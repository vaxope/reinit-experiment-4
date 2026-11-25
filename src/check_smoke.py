"""Check the rollout log of the smoke run: did slime really train, validate and push weights?

Purpose : the smoke run (scripts/deltaai/smoke_train.sh) runs real slime training for [smoke] steps
          (2). train.py exiting with status 0 is not enough: it must also have produced what the
          long run relies on. This check reads the answers src/reward.py logged and requires:
            1. every training step 1..steps logged, each with the full batch. Without dynamic
               sampling: exactly [rollout] global_batch_size answers (128) in rollout_batch_size
               groups (16) of n_samples_per_prompt (8). With DAPO dynamic sampling ([algorithm],
               2026-11-24) slime generates MORE and drops groups whose rewards are all equal, and
               the log has every answer that was scored (dropped groups too, and groups cut short
               when the batch filled). Then: at least rollout_batch_size complete groups WITH a
               learning signal (some answers right and some wrong: the rule of the correctness
               filter src/dapo_filters.py), i.e. enough groups to fill the batch;
            2. every answer labelled with a weight version, and step k generated with the weights
               after k - 1 updates: step 2 then proves that the trainer's update reached SGLang
               (the weight push), the one part a single step cannot show;
            3. validation before training (step 0) and after the last step, each answering all
               [smoke] eval_questions once;
            4. every reward = correctness ([reward] correct or incorrect) + overlong penalty
               (0 with [overlong] mode "none", else between -penalty_factor and 0);
            5. worst-case memory was exercised ([smoke] force_max_len_groups > 0): step 1 has a
               complete group with a signal (so it was trained on) that contains an answer of the
               full [rollout] max_response_len.
          It also prints the time each step took (from the log timestamps) and the answer lengths,
          as a first measurement for planning the long run.
Inputs  : runs/smoke_train_<job>/rollouts/ (rollouts_*.jsonl); configs/train_reinit.toml.
Output  : one line per check, then PASS or FAIL; exit code 1 on any failure. Reads only.
Pipeline: smoke_train.sh: slime train.py (2 steps) -> [check_smoke] -> pick_checkpoint -> convert
          -> check_trained -> serving test.
Usage   : python3 -m src.check_smoke runs/smoke_train_<job>/rollouts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pick_checkpoint import read_records, updates_done
from src.train_args import DEFAULT_TRAIN_CONFIG, load_train_config, smoke_config
from src.train_report import EXTRA, report


REWARD_PARTS = ("correctness_reward", "overlong_penalty")
TOL = 1e-9   # float tolerance for reward = correctness + penalty


def kept_groups(rs: list[dict], k: int) -> list[list[dict]]:
    """Complete groups (k answers) with some right and some wrong answers: what the correctness
    filter keeps (src/dapo_filters.check_correctness_varies)."""
    groups: dict = {}
    for r in rs:
        key = r.get("group_index") if r.get("group_index") is not None else r["qid"]
        groups.setdefault(key, []).append(r)
    return [g for g in groups.values() if len(g) == k and len({bool(r.get("correct")) for r in g}) > 1]


def signal_groups(rs: list[dict], k: int) -> int:
    return len(kept_groups(rs, k))


def reward_ok(r: dict, cfg: dict) -> bool:
    """reward = correctness + overlong penalty, each in its allowed range."""
    # read_records fills absent fields with None (logs written before 2026-11-24 have no parts)
    base = r.get("correctness_reward") if r.get("correctness_reward") is not None else r.get("reward")
    pen = r.get("overlong_penalty") or 0.0
    if base not in (cfg["reward"]["correct"], cfg["reward"]["incorrect"]) or r.get("reward") is None:
        return False
    low = 0.0 if cfg["overlong"]["mode"] == "none" else -cfg["overlong"]["penalty_factor"]
    return low - TOL <= pen <= TOL and abs(r["reward"] - (base + pen)) <= TOL


def check(rollout_dir: Path, cfg: dict) -> list[tuple[bool, str]]:
    """cfg = the REAL config; the smoke changes are applied here. Returns (ok, message) per check."""
    sc = smoke_config(cfg)
    steps, n_val = sc["run"]["num_steps"], cfg["smoke"]["eval_questions"]
    batch, k = sc["rollout"]["global_batch_size"], sc["rollout"]["n_samples_per_prompt"]
    n_groups = sc["rollout"]["rollout_batch_size"]
    dynamic = cfg["algorithm"]["dynamic_sampling"]
    recs = read_records(rollout_dir, EXTRA + REWARD_PARTS)
    out = []
    if not recs:
        return [(False, f"no answers logged in {rollout_dir}")]

    train = [r for r in recs if r["split"] != "eval"]
    unversioned = sum(updates_done(r) is None for r in recs)
    out.append((unversioned == 0, f"weight version on every answer ({unversioned} of {len(recs)} without)"))

    rep = report(rollout_dir, sc)
    got = sorted(rep["train"])
    out.append((got == list(range(1, steps + 1)), f"training steps logged {got} (from weight versions), "
                                                  f"expected 1..{steps}"))
    for s in range(1, steps + 1):
        st = rep["train"].get(s)
        n, ng, full = (st["n"], st["n_groups"], st["groups_full"]) if st else (0, 0, 0)
        if dynamic:
            step_recs = [r for r in train if updates_done(r) is not None and updates_done(r) + 1 == s]
            sig = signal_groups(step_recs, k)
            out.append((sig >= n_groups, f"step {s}: {n} answers in {ng} groups ({full} full, {sig} complete with "
                                         f"a learning signal), expected >= {n_groups} with a signal (dynamic sampling)"))
        else:
            ok = n == batch and ng == n_groups and full == n_groups
            out.append((ok, f"step {s}: {n} answers in {ng} groups ({full} full), "
                            f"expected {batch} in {n_groups} groups of {k}"))
    # (steps are numbered from the weight version, so "step 2 logged" with every answer versioned
    # means: 128 answers generated with the weights after 1 update, i.e. the push reached SGLang)

    val_steps = [(r["step"], r["n"]) for r in rep["validation"]]
    out.append((val_steps == [(0, n_val), (steps, n_val)],
                f"validation passes (step, answers) {val_steps}, expected [(0, {n_val}), ({steps}, {n_val})]"))

    n_force, limit = cfg["smoke"]["force_max_len_groups"], cfg["rollout"]["max_response_len"]
    if n_force > 0:
        step1 = [r for r in train if updates_done(r) == 0]
        full = sum(any((r.get("n_output_tokens") or 0) >= limit for r in g) for g in kept_groups(step1, k))
        out.append((full > 0, f"worst-case memory: {full} trained group(s) in step 1 with a {limit:,}-token answer "
                              f"(forced in the first {n_force} groups), expected >= 1"))

    bad = sum(not reward_ok(r, cfg) for r in recs)
    out.append((bad == 0, f"rewards = correctness {sorted({cfg['reward']['correct'], cfg['reward']['incorrect']})} "
                          f"+ overlong penalty ([overlong] mode {cfg['overlong']['mode']!r}) ({bad} other values)"))

    # timing and lengths: information for planning, never a failure (needs the versions)
    if unversioned or got != list(range(1, steps + 1)):
        return out
    ends = {s: max(r["time"] for r in train if updates_done(r) is not None and updates_done(r) + 1 == s)
            for s in got}
    first = min(r["time"] for r in recs)
    prev = first
    for s in got:
        st = rep["train"][s]
        print(f"[check_smoke] step {s}: done {ends[s] - prev:,.0f} s after the previous point; "
              f"answers median {st['tokens_median']:,.0f} / p90 {st['tokens_p90']:,.0f} tokens, "
              f"truncated {100 * st['truncation_rate']:.1f}%, mean reward {st['mean_reward']:.3f}")
        prev = ends[s]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rollouts", help="runs/smoke_train_<job>/rollouts")
    ap.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    a = ap.parse_args(argv)
    results = check(Path(a.rollouts), load_train_config(a.config))
    for ok, msg in results:
        print(f"[check_smoke] {'PASS' if ok else 'FAIL'} {msg}")
    ok = all(ok for ok, _ in results)
    print(f"[check_smoke] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
