"""DAPO dynamic-sampling filters on CORRECTNESS, loaded by slime (--dynamic-sampling-filter-path).

Purpose : DAPO drops the groups "with accuracy equal to 1 and 0" (all answers right or all wrong),
          because their group-normalised advantage is zero. slime's own filter
          (check_reward_nonzero_std) looks at the REWARD, and our reward includes the overlong
          penalty ([overlong] in configs/train_reinit.toml), so a group of 8 wrong answers whose
          lengths differ would be kept and train on length alone (review, 2026-11-24). These
          filters decide on correctness instead (user, 2026-11-24).
Inputs  : one group of slime Samples (the n_samples_per_prompt answers to one question), each
          already scored by src/reward.py, which sets `sample.gar_correct` (True/False) on the
          Sample object itself. A plain attribute, not sample.metadata, because the answers of a
          group may share one metadata dict; slime's Sample is a normal dataclass (no __slots__,
          slime/utils/types.py @ 79aa41a), and the filter receives the same objects reward scored
          (slime/rollout/sglang_rollout.py: generate_and_rm -> group -> call_dynamic_filter).
Output  : slime's DynamicFilterOutput(keep, reason): keep = the group has at least one right AND
          one wrong answer. reason is counted by slime's MetricGatherer
          (rollout/dynamic_filter/drop_<reason>) per step.
Pipeline: SGLang answers -> src/reward.py (reward + gar_correct) -> [dapo_filters] -> kept groups
          -> advantages -> training.
Fails   : a Sample without gar_correct raises instead of guessing: it means the reward did not run
          or another reward function is wired in.
          Stop rule (review 2026-11-27): slime keeps starting sampling rounds until the batch is
          full, with no round limit (sglang_rollout.py generate_rollout_async). If no group ever
          has a signal (wrong start weights or weight push: every answer unparsable; a broken
          prompt; a collapsed model), the job would spin until its time limit. After
          [algorithm] max_consecutive_dropped_groups dropped groups in a row, the filter raises,
          which ends train.py with the counts and the likely causes. One process runs all of
          slime's rollouts (the rollout manager), so a module-level counter sees every group.
"""
from __future__ import annotations

import os

ATTR = "gar_correct"   # set by src/reward.reward_one
_STREAK = {"n": 0, "reasons": {}, "limit": None}   # dropped groups in a row, by reason


def max_consecutive_drops() -> int:
    """[algorithm] max_consecutive_dropped_groups of the training config, read once per process."""
    if _STREAK["limit"] is None:
        from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config
        cfg = load_train_config(os.environ.get("GAR_TRAIN_CONFIG", DEFAULT_TRAIN_CONFIG))
        _STREAK["limit"] = cfg["algorithm"]["max_consecutive_dropped_groups"]
    return _STREAK["limit"]


def track_streak(keep: bool, reason: str | None) -> None:
    """Count dropped groups in a row; raise when the limit is reached (see the header)."""
    if keep:
        _STREAK["n"], _STREAK["reasons"] = 0, {}
        return
    _STREAK["n"] += 1
    _STREAK["reasons"][reason] = _STREAK["reasons"].get(reason, 0) + 1
    if _STREAK["n"] >= max_consecutive_drops():
        raise RuntimeError(
            f"dynamic sampling: {_STREAK['n']} groups in a row had no learning signal ({_STREAK['reasons']}); "
            "slime has no round limit, so the run would sample until its time limit. All 'all_wrong' usually "
            "means unusable answers: wrong start weights or weight push (check the first rollout sample in the "
            "log), a broken prompt, or a collapsed model. Limit: [algorithm] max_consecutive_dropped_groups.")


def _output(keep: bool, reason: str | None, fallback: bool):
    try:   # slime's result type; plain dict-like stand-in for local tests without slime
        from slime.rollout.filter_hub.base_types import DynamicFilterOutput
    except ImportError:
        from types import SimpleNamespace as DynamicFilterOutput   # noqa: N813
    out = DynamicFilterOutput(keep=keep, reason=reason)
    if fallback:
        out.keep_when_insufficient = True
    return out


def group_correctness(samples) -> list[bool]:
    flags = []
    for s in samples:
        s = s[-1] if isinstance(s, list) else s   # multi-segment trajectories: the last one is scored
        if not hasattr(s, ATTR):
            raise RuntimeError(f"sample has no {ATTR!r}: src/reward.py did not score it, so the "
                               "correctness filter cannot decide (check --custom-rm-path)")
        flags.append(bool(getattr(s, ATTR)))
    return flags


def _decide(samples):
    flags = group_correctness(samples)
    if all(flags):
        keep, reason = False, "all_correct"
    elif not any(flags):
        keep, reason = False, "all_wrong"
    else:
        keep, reason = True, None
    track_streak(keep, reason)
    return keep, reason


def check_correctness_varies(args, samples, **kwargs):
    """Keep the group only if it has a right and a wrong answer (DAPO dynamic sampling)."""
    keep, reason = _decide(samples)
    return _output(keep, reason, fallback=False)


def check_correctness_varies_with_fallback(args, samples, **kwargs):
    """Same decision, but slime keeps a rejected group rather than start another sampling round
    when the remaining candidates could not fill the batch (DynamicFilterOutput.keep_when_insufficient)."""
    keep, reason = _decide(samples)
    return _output(keep, reason, fallback=True)
