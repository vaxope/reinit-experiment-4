"""Unit tests for src/dapo_filters.py (DAPO dynamic sampling on correctness). Run: python3 tests/run_all.py

Fake slime Samples (types.SimpleNamespace); no slime needed. The key case: an all-wrong group
whose rewards differ only by the overlong penalty must be DROPPED (slime's reward-std filter
would keep it; review 2026-11-24).
"""
import asyncio
import os
import tempfile
from types import SimpleNamespace

from src import dapo_filters, reward


def group(*correct, rewards=None):
    rewards = rewards or [1.0 if c else 0.0 for c in correct]
    return [SimpleNamespace(gar_correct=c, reward=r) for c, r in zip(correct, rewards)]


def test_mixed_group_is_kept():
    out = dapo_filters.check_correctness_varies(None, group(True, False, False, True))
    assert out.keep is True and out.reason is None


def test_all_right_and_all_wrong_groups_are_dropped_with_a_reason():
    out = dapo_filters.check_correctness_varies(None, group(True, True, True))
    assert out.keep is False and out.reason == "all_correct"
    out = dapo_filters.check_correctness_varies(None, group(False, False, False))
    assert out.keep is False and out.reason == "all_wrong"


def test_all_wrong_with_different_overlong_penalties_is_dropped():
    """Rewards 0, -0.3, -1.0 differ (std > 0), but every answer is wrong: no correctness signal."""
    out = dapo_filters.check_correctness_varies(None, group(False, False, False, rewards=[0.0, -0.3, -1.0]))
    assert out.keep is False and out.reason == "all_wrong"


def test_fallback_variant_marks_keep_when_insufficient():
    out = dapo_filters.check_correctness_varies_with_fallback(None, group(False, False))
    assert out.keep is False and out.keep_when_insufficient is True
    assert getattr(dapo_filters.check_correctness_varies(None, group(False, False)), "keep_when_insufficient", False) is False


def test_unscored_sample_raises_instead_of_guessing():
    try:
        dapo_filters.check_correctness_varies(None, [SimpleNamespace(reward=1.0)])
    except RuntimeError as e:
        assert "gar_correct" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_multi_segment_trajectory_uses_its_last_segment():
    traj = [SimpleNamespace(), SimpleNamespace(gar_correct=True)]
    assert dapo_filters.check_correctness_varies(None, [traj, SimpleNamespace(gar_correct=False)]).keep is True


def test_reward_sets_the_attribute_the_filter_reads():
    """End to end: src/reward.py scores real-shaped samples, the filter decides on them."""
    d = tempfile.mkdtemp()
    old = os.environ.get("GAR_ROLLOUT_LOG_DIR")
    os.environ["GAR_ROLLOUT_LOG_DIR"] = d
    try:
        def s(text):
            return SimpleNamespace(response=text, label="B", status=SimpleNamespace(value="completed"),
                                   metadata={"qid": "syn1", "prompt_mode": "reinit"}, tokens=[1, 2, 3],
                                   response_length=3, rollout_id=None, index=0, group_index=0, weight_versions=["1"])
        right, wrong = s("x\n</analysis>\nanswer: B"), s("x\n</analysis>\nanswer: C")
        asyncio.run(reward.reward(None, right))
        asyncio.run(reward.reward(None, wrong))
        assert right.gar_correct is True and wrong.gar_correct is False
        assert dapo_filters.check_correctness_varies(None, [right, wrong]).keep is True
    finally:
        if old is None:
            os.environ.pop("GAR_ROLLOUT_LOG_DIR", None)
        else:
            os.environ["GAR_ROLLOUT_LOG_DIR"] = old
