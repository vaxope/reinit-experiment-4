"""Tests for src/check_smoke.py and pick_checkpoint --smoke on a synthetic smoke-run log (no model
output, no GPQA text). Run: python3 tests/run_all.py  (standard library only).

2026-11-24: DAPO dynamic sampling logs MORE than one batch per step (dropped groups too), and
overlong shaping makes rewards correctness + penalty; both are covered below."""
import copy
import json
import tempfile
from pathlib import Path

from src import check_smoke, pick_checkpoint
from src.train_args import load_train_config

CFG = load_train_config()
N_VAL = CFG["smoke"]["eval_questions"]
STEPS = CFG["smoke"]["steps"]
GROUPS, K = CFG["rollout"]["rollout_batch_size"], CFG["rollout"]["n_samples_per_prompt"]
L = CFG["rollout"]["max_response_len"]


def _rec(t, split, version, group, reward=1.0, penalty=0.0, n_tokens=None):
    return {"time": t, "split": split, "weight_versions": [] if version is None else [str(version)],
            "group_index": group, "qid": f"q{group}", "reward": reward + penalty, "correctness_reward": reward,
            "overlong_penalty": penalty, "correct": reward == 1.0,
            "lenient_correct": reward == 1.0, "truncated": False, "n_output_tokens": 1000 + group if n_tokens is None else n_tokens,
            "extracted_answer": "A", "analysis_closed": True}


def _log(steps=STEPS, val=N_VAL, versions=True, reward=1.0, dropped=0, groups=None, partial=0, penalty=0.0,
         full_len=True, wrong_dropped=0):
    """What a good smoke run logs: validation (weights after 0 updates), steps 1..n (step k with the
    weights after k-1 updates), validation after the last update. full_len: the forced answer of the
    worst-case memory test (step 1, group 0, its wrong answer) has the full response length."""
    v = (lambda x: x) if versions else (lambda x: None)
    recs, t = [], 0.0
    recs += [_rec(t + i, "eval", v(1), 1000 + i) for i in range(val)]; t += val
    for s in range(1, steps + 1):
        for g in range(GROUPS if groups is None else groups):   # kept groups: one answer wrong
            for j in range(K):
                n = L if (full_len and s == 1 and g == 0 and j == 0) else None
                recs.append(_rec(t, "train", v(s), s * 100 + g, reward if j else 0.0, penalty if j == 1 else 0.0, n)); t += 1
        for g in range(wrong_dropped):                         # all-wrong groups, rewards differ by length only
            for j in range(K):
                recs.append(_rec(t, "train", v(s), s * 100 + 60 + g, 0.0, -0.1 * j)); t += 1
        for g in range(dropped):                               # all-right groups dynamic sampling drops
            for j in range(K):
                recs.append(_rec(t, "train", v(s), s * 100 + 50 + g, 1.0)); t += 1
        for g in range(partial):                               # groups cut short when the batch filled
            for j in range(K // 2):
                recs.append(_rec(t, "train", v(s), s * 100 + 80 + g, float(j % 2))); t += 1
    recs += [_rec(t + i, "eval", v(STEPS + 1), 2000 + i) for i in range(val)]
    d = Path(tempfile.mkdtemp())
    with (d / "rollouts_host_1.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return d


def _ok(d, cfg=CFG):
    return all(ok for ok, _ in check_smoke.check(d, cfg))


def _cfg(**algorithm):
    c = copy.deepcopy(CFG)
    c["algorithm"].update(algorithm)
    return c


def test_good_smoke_log_passes():
    assert _ok(_log())


def test_failures_are_caught():
    assert not _ok(_log(steps=1))                 # step 2 (the weight push) missing
    assert not _ok(_log(versions=False))          # no weight versions: the push can't be proven
    assert not _ok(_log(val=N_VAL - 1))           # incomplete validation
    assert not _ok(_log(reward=0.5))              # a reward that is not 0/1
    assert not _ok(Path(tempfile.mkdtemp()))      # nothing logged


def test_pick_checkpoint_smoke_picks_the_saved_last_step():
    d = _log()
    ckpt = Path(tempfile.mkdtemp())
    (ckpt / f"iter_{STEPS - 1:07d}").mkdir()
    (ckpt / pick_checkpoint.TRACKER).write_text(str(STEPS - 1))
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pick_checkpoint.main(["--smoke", "--rollouts", str(d), "--ckpt-dir", str(ckpt), "--shell"])
    assert out.getvalue().strip() == f"STEP={STEPS} ITER_DIR={ckpt / f'iter_{STEPS - 1:07d}'}"
    try:   # without --smoke the real intervals (10) reject a pass at step 2
        pick_checkpoint.main(["--rollouts", str(d), "--ckpt-dir", str(ckpt), "--shell"])
    except (ValueError, SystemExit):
        pass
    else:
        raise AssertionError("expected refusal without --smoke")


def test_dynamic_sampling_extra_groups_pass_and_too_few_signal_groups_fail():
    dyn = _cfg(dynamic_sampling=True)
    assert _ok(_log(dropped=5, partial=2), dyn)                 # more than 128 answers is expected
    assert not _ok(_log(groups=GROUPS - 1, dropped=5), dyn)     # only 15 groups with a signal
    plain = _cfg(dynamic_sampling=False)
    assert _ok(_log(), plain)
    assert not _ok(_log(dropped=1), plain)                      # without it: exactly one batch


def test_overlong_penalty_in_rewards():
    c = copy.deepcopy(CFG)
    c["overlong"].update(mode="soft_punish", penalty_factor=1.0)
    assert _ok(_log(penalty=-0.4), c)                           # 1.0 - 0.4 = 0.6 is a valid reward
    assert not _ok(_log(penalty=-1.5), c)                       # beyond -penalty_factor
    c["overlong"]["mode"] = "none"
    assert not _ok(_log(penalty=-0.4), c)                       # no penalty allowed when off


def test_signal_is_correctness_not_reward():
    """All-wrong groups whose rewards differ by the overlong penalty are NOT counted as kept
    (they are what the correctness filter drops); 15 real groups + such groups must fail."""
    c = copy.deepcopy(CFG)
    c["overlong"].update(mode="soft_punish", penalty_factor=1.0)
    c["algorithm"]["dynamic_sampling"] = True
    assert not _ok(_log(groups=GROUPS - 1, wrong_dropped=3), c)
    assert _ok(_log(wrong_dropped=3), c)


def test_worst_case_memory_answer_is_required_when_forced():
    c = copy.deepcopy(CFG)
    c["smoke"]["force_max_len_groups"] = 32
    assert _ok(_log(full_len=True), c)
    assert not _ok(_log(full_len=False), c)     # nothing reached the full length: memory untested
    c["smoke"]["force_max_len_groups"] = 0
    assert _ok(_log(full_len=False), c)         # switched off: not required
