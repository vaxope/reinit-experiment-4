"""Unit tests for src/reward.py (DAPO reward: rule v3 + overlong shaping, per-rollout log). Run: python3 tests/run_all.py

Fake slime Samples (types.SimpleNamespace) with synthetic text; no slime, GPU or GPQA needed.
"""
import asyncio
import copy
import enum
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src import reward


class Status(enum.Enum):          # mirrors slime's Sample.Status values
    COMPLETED = "completed"
    TRUNCATED = "truncated"


def sample(response, label="B", status=Status.COMPLETED, mode="reinit", split=None, n=5):
    meta = {"qid": "syn0001", "prompt_mode": mode}
    if split:
        meta["_split"] = split
    return SimpleNamespace(response=response, label=label, status=status, metadata=meta,
                           tokens=list(range(100, 120)), response_length=n, rollout_id=3,
                           index=7, group_index=1, weight_versions=["3"])


def run(s):
    return asyncio.run(reward.reward(None, s))


def with_log_dir(fn):
    def wrapper():
        d = tempfile.mkdtemp()
        old = os.environ.get("GAR_ROLLOUT_LOG_DIR")
        os.environ["GAR_ROLLOUT_LOG_DIR"] = d
        try:
            return fn(Path(d))
        finally:
            if old is None:
                os.environ.pop("GAR_ROLLOUT_LOG_DIR", None)
            else:
                os.environ["GAR_ROLLOUT_LOG_DIR"] = old
    wrapper.__name__ = fn.__name__
    return wrapper


def logged(d):
    return [json.loads(l) for f in sorted(d.glob("rollouts_*.jsonl")) for l in f.read_text().splitlines()]


@with_log_dir
def test_correct_answer_after_analysis_gets_1(d):
    assert run(sample("steps...\n</analysis>\n\nanswer: B<|im_end|>")) == 1.0
    rec = logged(d)[0]
    assert rec["correct"] and rec["extracted_answer"] == "B" and rec["reward"] == 1.0
    assert rec["qid"] == "syn0001" and rec["gold"] == "B" and rec["split"] == "train"


@with_log_dir
def test_wrong_letter_and_no_answer_get_0(d):
    assert run(sample("x\n</analysis>\n\nanswer: C")) == 0.0
    assert run(sample("x\n</analysis>\n\nI am not sure.")) == 0.0


@with_log_dir
def test_analysis_never_closed_gets_0_even_if_lenient_finds_the_right_letter(d):
    # the control run's most common reinit failure: closed with </think> instead of </analysis>
    assert run(sample("work\nAnswer: B\n</think>\n\nanswer: B")) == 0.0
    rec = logged(d)[0]
    assert rec["extraction_rule"] == "unclosed_analysis" and rec["lenient_correct"] is True


@with_log_dir
def test_cut_off_answer_gets_0_even_with_the_right_letter(d):
    assert run(sample("x\n</analysis>\nanswer: B", status=Status.TRUNCATED)) == 0.0
    rec = logged(d)[0]
    assert rec["truncated"] and rec["status"] == "truncated" and not rec["lenient_correct"]


@with_log_dir
def test_log_has_the_answer_tokens_and_split(d):
    run(sample("x\n</analysis>\nanswer: B", split="eval", n=5))
    rec = logged(d)[0]
    assert rec["output_token_ids"] == list(range(115, 120))      # last response_length tokens
    assert rec["split"] == "eval" and rec["rollout_id"] == 3 and rec["n_output_tokens"] == 5
    assert rec["reward_rule"] == "last_answer_field_v3"


@with_log_dir
def test_list_input_and_default_mode(d):
    out = run([sample("x\n</analysis>\nanswer: B"), sample("x\n</analysis>\nanswer: A")])
    assert out == [1.0, 0.0] and len(logged(d)) == 2
    # the matched control (default mode): whole output is visible, no <analysis> needed
    assert run(sample("reasoning...\nanswer: B", mode="default")) == 1.0


def test_missing_log_dir_stops_the_run():
    old = os.environ.pop("GAR_ROLLOUT_LOG_DIR", None)
    try:
        run(sample("x\n</analysis>\nanswer: B"))
    except RuntimeError:
        return
    finally:
        if old is not None:
            os.environ["GAR_ROLLOUT_LOG_DIR"] = old
    raise AssertionError("expected RuntimeError without GAR_ROLLOUT_LOG_DIR")


# ---------------------------------------------------------------- DAPO overlong shaping (2026-11-24)
CFG = reward.config()
L = CFG["rollout"]["max_response_len"]


def _cfg(mode="soft_punish", buf=4096, factor=1.0):
    c = copy.deepcopy(CFG)
    c["overlong"].update(mode=mode, buffer_len=buf, penalty_factor=factor)
    return c


def test_overlong_penalty_shape():
    c = _cfg(buf=4096, factor=1.0)
    start = L - 4096
    assert reward.overlong_penalty(0, c) == 0.0
    assert reward.overlong_penalty(start, c) == 0.0                 # up to L - B: no penalty
    assert reward.overlong_penalty(start + 2048, c) == -0.5         # linear inside the buffer
    assert reward.overlong_penalty(L, c) == -1.0                    # at the limit: -factor
    assert reward.overlong_penalty(L + 100, c) == -1.0              # clamped
    assert reward.overlong_penalty(start + 1024, _cfg(factor=2.0)) == -0.5
    assert reward.overlong_penalty(L, _cfg(mode="none")) == 0.0     # switched off


def test_overlong_values_come_from_the_config_not_the_code():
    """The DAPO values are placeholders: changing the config must change the penalty."""
    a = reward.overlong_penalty(L - 100, _cfg(buf=1000))
    b = reward.overlong_penalty(L - 100, _cfg(buf=8000))
    assert abs(a - (-900 / 1000)) < 1e-12      # 900 tokens into a 1,000-token buffer
    assert abs(b - (-7900 / 8000)) < 1e-12     # 7,900 tokens into an 8,000-token buffer


def test_score_adds_the_penalty_and_keeps_both_parts():
    c = _cfg(buf=4096, factor=1.0)
    text = "x\n</analysis>\nanswer: B"
    right_long = reward.score(text, "B", "reinit", False, c, n_tokens=L - 1024)
    assert right_long["correctness_reward"] == 1.0 and right_long["overlong_penalty"] == -0.75
    assert right_long["reward"] == 0.25 and right_long["correct"] is True       # accuracy still readable
    cut = reward.score(text, "B", "reinit", True, c, n_tokens=L)
    assert cut["correctness_reward"] == 0.0 and cut["reward"] == -1.0 and not cut["correct"]
    short = reward.score(text, "B", "reinit", False, c, n_tokens=500)
    assert short["reward"] == 1.0 and short["overlong_penalty"] == 0.0


@with_log_dir
def test_sample_length_reaches_the_penalty_and_the_log(d):
    if CFG["overlong"]["mode"] != "soft_punish":
        return "SKIP: [overlong] mode is not soft_punish in the current config"
    r = run(sample("x\n</analysis>\nanswer: B", status=Status.TRUNCATED, n=L))
    rec = logged(d)[0]
    assert r == rec["reward"] == CFG["reward"]["incorrect"] - CFG["overlong"]["penalty_factor"]
    assert rec["overlong_penalty"] == -CFG["overlong"]["penalty_factor"] and rec["correctness_reward"] == 0.0


@with_log_dir
def test_reward_marks_correctness_on_the_sample_for_the_filter(d):
    right, wrong = sample("x\n</analysis>\nanswer: B"), sample("x\n</analysis>\nanswer: C")
    cut = sample("x\n</analysis>\nanswer: B", status=Status.TRUNCATED)
    for x in (right, wrong, cut):
        run(x)
    assert right.gar_correct is True and wrong.gar_correct is False and cut.gar_correct is False
