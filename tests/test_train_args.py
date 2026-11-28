"""Unit tests for src/train_args.py (training config -> slime flags). Run: python3 tests/run_all.py

They check that the parameter sheet reaches slime unchanged, and that every guard refuses what it
should:
  - DAPO (user, 2026-11-22): clip-higher, token-level loss, dynamic sampling, KL switch; every
    value is a placeholder, so the tests also prove that changing a value in the config changes
    the flag (nothing hard-coded);
  - separate GPUs (user, 2026-11-22): 2 training + 2 SGLang on one node, never --colocate or offload;
  - Qwen3.5-4B architecture vs its config.json (values fetched 2026-11-24);
  - sampling / reward / model differing from the control config.
The control-match tests use the real 4B control config (configs/control.toml, 2026-11-27) and, to
prove the model check refuses a mismatch, a 2B stand-in: that config with ONLY [model] set back to
the 2B of control_v1. No slime, GPU or network needed.
"""
import copy
import json
import re
import tempfile
from pathlib import Path

from src import train_args
from src.config import REPO_ROOT, load_config
from src.train_args import (REQUIRED, FORBIDDEN_FLAGS, HF_CONFIG_KEYS, PARSED_NAME, _flag_value, build_args,
                            check_against_control, check_algorithm, check_all, check_flags, check_hardware,
                            check_hf_config, disk_estimate_gb, expected_resolved, load_control_for,
                            load_train_config, main, memory_estimate_gb, model_args, smoke_config)

CFG = load_train_config()
CONTROL = load_config()
PATHS = dict(hf_checkpoint="HF", ref_load="REF", save="SAVE", prompt_data="P.jsonl", eval_data="E.jsonl")

# Qwen3.5-4B @ 851bf6e8, config.json text_config (fetched 2026-11-24): the numbers the config must hold.
QWEN35_4B = {"num_hidden_layers": 32, "hidden_size": 2560, "intermediate_size": 9216,
             "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
             "rope_theta": 10000000, "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6,
             "max_position_embeddings": 262144}


def control_2b_stand_in() -> dict:
    """configs/control.toml with only [model] set to control_v1's 2B (see the module doc)."""
    c = copy.deepcopy(CONTROL)
    c["model"].update(name="Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
                      chat_template_sha256="273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80")
    return c


CONTROL_2B = control_2b_stand_in()


def value(flags, flag):
    """The value that follows `flag` (one occurrence expected)."""
    assert flags.count(flag) == 1, (flag, flags.count(flag))
    return flags[flags.index(flag) + 1]


def raises(fn, *a):
    try:
        fn(*a)
    except (ValueError, KeyError, FileNotFoundError):
        return True
    return False


def changed(section, **kv):
    c = copy.deepcopy(CFG)
    c[section].update(kv)
    return c


# ---------------------------------------------------------------- the real config
def test_real_config_passes_all_checks_against_a_matching_control():
    check_all(CFG, CONTROL)


def test_training_targets_qwen35_4b_and_refuses_the_2b_control():
    """User, 2026-11-24: 4B only. Against the existing 2B control config the model check must refuse."""
    assert CFG["model"]["hf_repo"] == "Qwen/Qwen3.5-4B"
    assert raises(check_against_control, CFG, CONTROL_2B)


def test_missing_control_config_is_refused_with_a_clear_message():
    """[run] control_config names the 4B control; if it is missing, nothing can be built.
    Tested with a path that never exists."""
    missing = str(Path(tempfile.mkdtemp()) / "control_missing.toml")
    try:
        load_control_for(CFG, missing)
    except FileNotFoundError as e:
        assert "control_config" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
    try:
        main(["--hf-checkpoint", tempfile.mkdtemp(), "--check", "--control-config", missing])
    except SystemExit as e:
        assert "control config" in str(e)
    else:
        raise AssertionError("expected the CLI to stop")


def test_check_mode_passes_with_a_matching_control_and_config_json():
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({"text_config": QWEN35_4B}))
    ctl = d / "control_copy.toml"
    src = (REPO_ROOT / "configs" / "control.toml").read_text()
    for key, new in (("name", CFG["model"]["hf_repo"]), ("revision", CFG["model"]["revision"]),
                     ("chat_template_sha256", CFG["model"]["chat_template_sha256"])):
        src, n = re.subn(rf'(?m)^{key} = ".*"$', f'{key} = "{new}"', src, count=1)
        assert n == 1, key
    ctl.write_text(src)
    real_check_data = train_args.check_data
    train_args.check_data = lambda cfg, out_dir: None   # the data files are tested on their own below
    try:
        main(["--hf-checkpoint", str(d), "--check", "--control-config", str(ctl)])   # no exception = pass
    finally:
        train_args.check_data = real_check_data


# ---------------------------------------------------------------- the slime input files (review 2026-11-24)
def _built_data(**manifest_changes):
    """A fake train_prep output folder whose manifest matches the current prompts and config."""
    from src import prompts
    from src.train_data import sha256_file
    d = Path(tempfile.mkdtemp())
    for split in ("train", "val"):
        (d / f"{split}.jsonl").write_text(json.dumps({"text": "x", "label": "A"}) + "\n")
    man = {"prompt_hash": prompts.PROMPT_HASH, "prompt_mode": CFG["data"]["prompt_mode"],
           "shuffle_salt": CFG["data"]["shuffle_salt"],
           "checks": {"prompt_tokens_max": 1000, "max_prompt_len": CFG["rollout"]["max_prompt_len"]},
           "outputs_sha256": {s: sha256_file(d / f"{s}.jsonl") for s in ("train", "val")}}
    man.update(manifest_changes)
    (d / "manifest.json").write_text(json.dumps(man))
    return d


def test_data_built_from_the_current_prompts_passes():
    train_args.check_data(CFG, _built_data())


def test_data_built_from_an_older_prompt_is_refused():
    """The final prompts will change src/prompts.py: training must not run on data built before."""
    assert raises(train_args.check_data, CFG, _built_data(prompt_hash="0000000000000000"))
    assert raises(train_args.check_data, CFG, _built_data(prompt_mode="default"))
    assert raises(train_args.check_data, CFG, _built_data(shuffle_salt="other"))


def test_prompts_longer_than_the_cap_are_refused():
    """Few-shot examples lengthen the prompt; slime would drop over-long prompts silently."""
    long = {"prompt_tokens_max": CFG["rollout"]["max_prompt_len"] + 1}
    assert raises(train_args.check_data, CFG, _built_data(checks=long))


def test_half_copied_or_missing_data_is_refused():
    d = _built_data()
    (d / "val.jsonl").write_text("")                     # truncated copy
    assert raises(train_args.check_data, CFG, d)
    d = _built_data()
    (d / "manifest.json").unlink()
    assert raises(train_args.check_data, CFG, d)


def test_previous_experiment_values_are_kept():
    """User, 2026-11-24: everything except DAPO, the GPU layout and the model as in Experiment-3."""
    f = build_args(CFG, PATHS)
    expected = {
        "--n-samples-per-prompt": "8", "--rollout-batch-size": "16", "--global-batch-size": "128",
        "--rollout-max-response-len": "32768", "--eval-max-response-len": "32768",
        "--rollout-max-prompt-len": "2304",   # user, 2026-11-27: raised from 2,048 for the few-shot prompts
        "--optimizer": "adam", "--lr": "1e-06", "--lr-decay-style": "constant", "--lr-warmup-iters": "0",
        "--adam-beta1": "0.9", "--adam-beta2": "0.999", "--adam-eps": "1e-08",
        "--weight-decay": "0.01", "--clip-grad": "1.0",
        "--rollout-temperature": "1.0", "--rollout-top-p": "0.95", "--rollout-top-k": "20",
        "--eval-temperature": "1.0", "--eval-top-p": "0.95", "--eval-top-k": "20",
        "--attention-dropout": "0.0", "--hidden-dropout": "0.0", "--seed": "1234",
        # 25, not Experiment-3's 10: changed on purpose for the 500 GB quota (user, 2026-11-24)
        "--save-interval": "25", "--eval-interval": "25", "--num-rollout": "100",
        "--max-tokens-per-gpu": "36864", "--log-probs-chunk-size": "1024", "--sglang-mem-fraction-static": "0.7",
    }
    for flag, want in expected.items():
        assert value(f, flag) == want, (flag, value(f, flag), want)
    for switch in ("--rollout-shuffle", "--use-dynamic-batch-size", "--accumulate-allreduce-grads-in-fp32",
                   "--attention-softmax-in-fp32", "--no-gradient-accumulation-fusion"):
        assert switch in f, switch


# ---------------------------------------------------------------- DAPO
def test_dapo_flags_reach_slime():
    f = build_args(CFG, PATHS)
    al = CFG["algorithm"]
    assert value(f, "--advantage-estimator") == "grpo"
    assert value(f, "--eps-clip") == _flag_value(al["eps_clip"])
    assert value(f, "--eps-clip-high") == _flag_value(al["eps_clip_high"])
    assert ("--calculate-per-token-loss" in f) == al["per_token_loss"]
    if al["dynamic_sampling"]:
        assert value(f, "--dynamic-sampling-filter-path") == al["dynamic_sampling_filter"]
        assert value(f, "--over-sampling-batch-size") == str(al["over_sampling_batch_size"])
    assert ("--use-kl-loss" in f) == al["use_kl_loss"]
    assert "--disable-grpo-std-normalization" not in f and "--use-tis" not in f


def test_dapo_values_are_read_from_the_config_not_hard_coded():
    """The DAPO values are placeholders: each one must follow the config when it is changed."""
    c = changed("algorithm", eps_clip=0.1, eps_clip_high=0.35, over_sampling_batch_size=48)
    f = build_args(c, PATHS)
    assert value(f, "--eps-clip") == "0.1" and value(f, "--eps-clip-high") == "0.35"
    assert value(f, "--over-sampling-batch-size") == "48"
    off = changed("algorithm", per_token_loss=False, dynamic_sampling=False)
    f = build_args(off, PATHS)
    assert "--calculate-per-token-loss" not in f
    assert "--dynamic-sampling-filter-path" not in f and "--over-sampling-batch-size" not in f
    kl = changed("algorithm", use_kl_loss=True, kl_loss_coef=0.001)
    check_algorithm(kl)
    f = build_args(kl, PATHS)
    assert "--use-kl-loss" in f and value(f, "--kl-loss-coef") == "0.001" and value(f, "--kl-loss-type") == "k3"
    fallback = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std_with_fallback"
    assert value(build_args(changed("algorithm", dynamic_sampling_filter=fallback), PATHS),
                 "--dynamic-sampling-filter-path") == fallback



def test_drop_limit_below_one_sampling_round_is_refused():
    """max_consecutive_dropped_groups < over_sampling_batch_size would stop a healthy first round."""
    bad = changed("algorithm", max_consecutive_dropped_groups=CFG["algorithm"]["over_sampling_batch_size"] - 1)
    try:
        check_algorithm(bad)
    except ValueError as e:
        assert "max_consecutive_dropped_groups" in str(e)
    else:
        raise AssertionError("expected ValueError")

def test_inconsistent_dapo_settings_are_refused():
    bad = [("algorithm", dict(eps_clip_high=0.1)),                        # upper below lower
           ("algorithm", dict(eps_clip=0.0)),
           ("algorithm", dict(dynamic_sampling_filter="")),
           ("algorithm", dict(over_sampling_batch_size=8)),               # < 16 questions per step
           ("algorithm", dict(use_kl_loss=True, kl_loss_coef=0.0)),
           ("algorithm", dict(use_kl_loss=False, kl_loss_coef=0.001)),    # a KL value that would be ignored
           ("algorithm", dict(use_tis=True)), ("algorithm", dict(entropy_coef=0.01)),
           ("algorithm", dict(std_normalization=False)), ("algorithm", dict(advantage_estimator="gspo")),
           ("algorithm", dict(name="grpo")),
           ("rollout", dict(global_batch_size=256)),                      # not one update per step
           ("eval", dict(max_response_len=16384)),
           ("overlong", dict(mode="filter")), ("overlong", dict(buffer_len=0)),
           ("overlong", dict(buffer_len=32768)), ("overlong", dict(penalty_factor=0.0)),
           ("data", dict(prompt_mode="default"))]
    for section, kv in bad:
        assert raises(check_algorithm, changed(section, **kv)), (section, kv)
    check_algorithm(changed("overlong", mode="none", buffer_len=0))     # buffer unused when off


def test_expected_resolved_follows_dapo_switches():
    exp = expected_resolved(CFG)
    al = CFG["algorithm"]
    assert exp["eps_clip_high"] == al["eps_clip_high"] and exp["calculate_per_token_loss"] == al["per_token_loss"]
    assert exp["use_tis"] is False and exp["grpo_std_normalization"] is True
    off = expected_resolved(changed("algorithm", dynamic_sampling=False))
    assert off["dynamic_sampling_filter_path"] is None and "over_sampling_batch_size" not in off


# ---------------------------------------------------------------- separate GPUs
def test_training_and_generation_get_separate_gpus():
    f = build_args(CFG, PATHS)
    hw = CFG["hardware"]
    assert (hw["actor_num_gpus"], hw["rollout_num_gpus"], hw["gpus_per_node"]) == (2, 2, 4)   # user, 2026-11-22
    assert value(f, "--actor-num-gpus-per-node") == "2" and value(f, "--rollout-num-gpus") == "2"
    assert value(f, "--num-gpus-per-node") == "4" and value(f, "--actor-num-nodes") == "1"
    assert "--use-distributed-optimizer" in f
    assert value(f, "--tensor-model-parallel-size") == "2" and "--sequence-parallel" in f   # user, 2026-11-24
    assert value(f, "--context-parallel-size") == "1" and value(f, "--pipeline-model-parallel-size") == "1"
    for flag in ("--colocate", "--offload", "--offload-train", "--offload-rollout", "--rollout-function-path"):
        assert flag not in f, flag
    exp = expected_resolved(CFG)
    assert exp["colocate"] is False and exp["offload_train"] is False and exp["offload_rollout"] is False


def test_bad_gpu_layouts_are_refused():
    for kv in (dict(actor_num_gpus=3), dict(rollout_num_gpus=0, actor_num_gpus=4),
               dict(rollout_num_gpus_per_engine=3), dict(tensor_model_parallel_size=3),
               dict(rollout_mode="fully_async"), dict(rollout_mode="colocate"),
               dict(tensor_model_parallel_size=1, sequence_parallel=True),     # Megatron would drop SP
               dict(update_weight_transport="tcp")):
        assert raises(check_hardware, changed("hardware", **kv)), kv
    m = copy.deepcopy(CFG); m["megatron_model"]["num_query_groups"] = 3
    assert raises(check_hardware, m)                                   # TP 2 must divide the key/value groups


# ---------------------------------------------------------------- memory, offload, disk (review 2026-11-24)
def test_memory_estimate_fits_with_tp2_and_offload_and_refuses_what_would_oom():
    mem = memory_estimate_gb(CFG)
    assert mem["total"] < mem["limit"], mem
    # the review's case: tensor parallel 1 puts all 248,320 vocabulary logits (fp32 + grads) on one GPU
    tp1 = changed("hardware", tensor_model_parallel_size=1, sequence_parallel=False)
    assert memory_estimate_gb(tp1)["total"] > mem["limit"] and raises(check_hardware, tp1)
    # tensor parallel 2 but the optimizer on the GPU: ~105 GB, refused
    no_off = changed("hardware", optimizer_cpu_offload=False)
    assert raises(check_hardware, no_off)
    # logits: 36,864 tokens x 124,160 (half vocabulary) x 10 bytes
    assert abs(mem["logits"] - 36864 * 248320 / 2 * 10 / 1e9) < 1e-6


def test_optimizer_offload_flags_are_paired_as_megatron_requires():
    f = build_args(CFG, PATHS)
    for flag in ("--optimizer-cpu-offload", "--use-precision-aware-optimizer", "--use-distributed-optimizer"):
        assert flag in f, flag
    assert value(f, "--optimizer-offload-fraction") == "1.0"
    for flag in ("--main-params-dtype", "--main-grads-dtype", "--exp-avg-dtype", "--exp-avg-sq-dtype"):
        assert value(f, flag) == "fp32", flag
    for kv in (dict(use_precision_aware_optimizer=False), dict(use_distributed_optimizer=False),
               dict(exp_avg_dtype="bf16"), dict(main_grads_dtype="bf16"), dict(optimizer_offload_fraction=0.5)):
        assert raises(check_hardware, changed("hardware", **kv)), kv
    exp = expected_resolved(CFG)
    assert exp["optimizer_cpu_offload"] is True and exp["main_grads_dtype"] == "fp32"


def test_one_full_sequence_must_fit_a_micro_batch():
    assert raises(check_hardware, changed("hardware", max_tokens_per_gpu=32768))
    longer = changed("rollout", max_prompt_len=8192)                 # few-shot prompts: prompt cap raised alone
    assert raises(check_hardware, longer)
    check_hardware(changed("rollout", max_prompt_len=4096))          # 4,096 + 32,768 = 36,864: still fits


def test_checkpoints_fit_the_disk_budget():
    assert disk_estimate_gb(CFG) == 4 * CFG["checkpoint"]["est_save_gb"] <= CFG["checkpoint"]["disk_budget_gb"]
    assert raises(check_hardware, changed("checkpoint", save_interval=10))   # Experiment-3's 10: ~600 GB
    odd = changed("checkpoint", save_interval=30)                             # 30, 60, 90 + the last step
    assert disk_estimate_gb(odd) == 4 * CFG["checkpoint"]["est_save_gb"]


def test_weight_push_transport_and_disk_fallback():
    f = build_args(CFG, PATHS)
    assert value(f, "--update-weight-transport") == "nccl" and "--update-weight-disk-dir" not in f
    disk = changed("hardware", update_weight_transport="disk")
    assert raises(build_args, disk, PATHS)                              # needs a folder
    f = build_args(disk, dict(PATHS, update_weight_disk_dir="WSYNC"))
    assert value(f, "--update-weight-transport") == "disk" and value(f, "--update-weight-disk-dir") == "WSYNC"


def test_reward_std_filter_is_refused_while_overlong_shaping_is_on():
    slime_filter = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    assert CFG["algorithm"]["dynamic_sampling_filter"] == "src.dapo_filters.check_correctness_varies"
    assert raises(check_algorithm, changed("algorithm", dynamic_sampling_filter=slime_filter))
    off = changed("algorithm", dynamic_sampling_filter=slime_filter)
    off["overlong"]["mode"] = "none"
    check_algorithm(off)                                                # rewards are pure correctness then


def test_gpu_layout_values_are_read_from_the_config():
    c = changed("hardware", actor_num_gpus=1, rollout_num_gpus=3, tensor_model_parallel_size=1,
                sequence_parallel=False)
    f = build_args(c, PATHS)
    assert value(f, "--actor-num-gpus-per-node") == "1" and value(f, "--rollout-num-gpus") == "3"


def test_no_forbidden_flag_is_ever_emitted():
    f = build_args(CFG, PATHS)
    assert not set(f) & set(FORBIDDEN_FLAGS)
    for bad in FORBIDDEN_FLAGS:
        assert raises(check_flags, ["--lr", "1e-6", bad])


def test_smoke_forcing_is_only_set_by_the_smoke_script():
    """$GAR_SMOKE_FORCE_MAX_LEN_GROUPS changes sampling; only smoke_train.sh may set it."""
    for f in (REPO_ROOT / "scripts" / "deltaai").iterdir():
        if f.suffix in (".sh", ".sbatch") and f.name != "smoke_train.sh":
            assert "GAR_SMOKE_FORCE_MAX_LEN_GROUPS" not in f.read_text(), f.name
    assert CFG["smoke"]["force_max_len_groups"] <= CFG["algorithm"]["over_sampling_batch_size"]


def test_nccl_and_gloo_stay_on_loopback():
    text = (REPO_ROOT / "scripts" / "deltaai" / "train_inside.sh").read_text()
    for line in ("export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0", "export SLIME_HOST_IP=127.0.0.1",
                 "export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo"):
        assert line in text, line
    start = text.index('timeout "$RAY_START_TIMEOUT_S" ray start')        # the command, not the header
    assert text.index("export NCCL_SOCKET_IFNAME") < start             # set before Ray starts the workers


def test_job_scripts_request_the_configs_gpu_count():
    """Slurm reads #SBATCH lines before any code runs, so the number is written in the script;
    it must equal [hardware] gpus_per_node (submit.sh reads the same line for the cost)."""
    want = CFG["hardware"]["gpus_per_node"]
    d = REPO_ROOT / "scripts" / "deltaai"

    def sbatch(name, opt):
        return re.findall(rf"(?m)^#SBATCH --{opt}=(\S+)$", (d / name).read_text())

    for name in ("slime_setup.sbatch", "train_reinit.sbatch", "setup_and_train.sbatch", "hold_node.sbatch"):
        assert sbatch(name, "gpus-per-node") == [str(want)], name
        for opt in ("cpus-per-task", "mem"):   # setup + training in one job: the same resources
            assert sbatch(name, opt) == sbatch("slime_setup.sbatch", opt), (name, opt)
    for name in ("smoke_train.sh", "train_reinit.sbatch"):
        assert 'cfg_value "$CFG" hardware gpus_per_node' in (d / name).read_text(), name   # Ray gets the same number


def test_jobs_that_serve_the_control_model_request_dp_size_gpus():
    """SGLang starts [server] dp_size model copies, one per GPU (user, 2026-11-27): the control jobs and
    the standalone evaluation request exactly that many GPUs; the training node must hold them, since
    the evaluation also runs inside the training job (train_args.check_against_control)."""
    dp = CONTROL["server"]["dp_size"]
    d = REPO_ROOT / "scripts" / "deltaai"
    for name in ("run_control.sbatch", "smoke_test.sbatch", "eval_trained.sbatch"):
        got = re.findall(r"(?m)^#SBATCH --gpus-per-node=(\d+)$", (d / name).read_text())
        assert got == [str(dp)], (name, got)
    assert '--dp-size "$DP_SIZE"' in (d / "serve_and_generate.sh").read_text()
    assert raises(check_against_control, CFG, dict(CONTROL, server=dict(CONTROL["server"], dp_size=want_too_many())))


def want_too_many():
    return CFG["hardware"]["gpus_per_node"] + 1


def test_training_job_checks_flags_and_evaluates_with_the_training_control_config():
    """train_reinit.sbatch: slime's parser on this job's flags with training's CUDA_DEVICE_MAX_CONNECTIONS
    (setup_pass no longer implies the flag check), the time budget from [job], and the evaluation
    with the control config of [run] control_config (not the default file)."""
    d = REPO_ROOT / "scripts" / "deltaai"
    train = (d / "train_reinit.sbatch").read_text()
    inside = re.search(r"(?m)^export .*CUDA_DEVICE_MAX_CONNECTIONS=(\S+)", (d / "train_inside.sh").read_text())
    cmd = train.index("python3 scripts/deltaai/check_slime_args.py")   # the command, not the header
    check = train[train.rindex("\nS env", 0, cmd):cmd]
    assert f"CUDA_DEVICE_MAX_CONNECTIONS={inside.group(1)}" in check
    assert cmd < train.index("bash scripts/deltaai/train_inside.sh")   # checked before training
    both_text = (d / "setup_and_train.sbatch").read_text()
    for key in REQUIRED["job"]:
        assert f"job {key}" in train + both_text, key   # every [job] value is read by one of the two jobs
    # a resumed run must have the flags and data it started with (run_identity.txt), checked before training
    ident = train.index('ID_FILE="$RUN_DIR/run_identity.txt"')
    assert train.index("--out \"$FLAGS_FILE\"") < ident < train.index("bash scripts/deltaai/train_inside.sh")
    assert 'sha256sum "$DATA_DIR/manifest.json"' in train
    # submit.sh's marker preflight builds the same folder name as slime_paths (<model>_<rev 8>_torch_dist)
    common = (d / "slime_common.sh").read_text()
    assert 'REF_DIR="$SLIME_CKPT_DIR/$(basename "$repo")_${rev:0:8}_torch_dist"' in common
    assert 'ref="$SLIME_CKPT_DIR/${repo##*/}_${rev:0:8}_torch_dist"' in (d / "submit.sh").read_text()
    assert train.rstrip().endswith("bash scripts/deltaai/eval_trained.sbatch best")
    ev = (d / "eval_trained.sbatch").read_text()
    assert 'cfg_value "$CFG" run control_config' in ev and "export CONTROL_CONFIG" in ev
    assert ev.index("export CONTROL_CONFIG") < ev.index("serve_and_generate.sh --run-id")
    both = both_text
    setup, ctl, trn = (both.index("bash scripts/deltaai/slime_setup.sbatch"),
                       both.index("bash scripts/deltaai/serve_and_generate.sh --run-id \"$CTL_ID\""),
                       both.index("bash scripts/deltaai/train_reinit.sbatch"))
    assert setup < ctl < trn                      # setup + smoke, then the control run, then training
    # the control run is capped and never stops the job (set +e around it, set -e after)
    assert both.rindex("set +e", 0, ctl) > setup and both.index("set -e", ctl) < trn
    assert 'timeout --signal=TERM --kill-after=120 "$((CTL_MAX_MIN * 60))"' in both
    # the smoke run is repeated when code/config changed since the commit its marker records
    assert 'git diff --quiet "$SMOKE_GIT" HEAD -- src configs scripts/deltaai' in both
    sub = (d / "submit.sh").read_text()
    for kind, script in (("train", "train_reinit.sbatch"), ("setup-train", "setup_and_train.sbatch"),
                         ("eval-trained", "eval_trained.sbatch")):
        assert re.search(rf"(?m)^  {kind}\) .*SCRIPT=scripts/deltaai/{script};", sub), kind
        assert (d / script).exists()


# ---------------------------------------------------------------- unchanged machinery
def test_rollout_attention_kernels_are_pinned_to_the_controls():
    f = build_args(CFG, PATHS)
    assert value(f, "--sglang-attention-backend") == CONTROL["server"]["attention_backend"] == "flashinfer"
    assert value(f, "--sglang-mm-attention-backend") == CONTROL["server"]["mm_attention_backend"] == "triton_attn"
    assert raises(check_against_control, changed("hardware", sglang_attention_backend="fa3"), CONTROL)


def test_gradient_accumulation_fusion_off_in_conversion_and_training():
    for flags in (model_args(CFG), build_args(CFG, PATHS)):
        assert flags.count("--no-gradient-accumulation-fusion") == 1
    assert expected_resolved(CFG)["gradient_accumulation_fusion"] is False


def test_norm_epsilon_is_checked_under_megatrons_attribute_name():
    exp = expected_resolved(CFG)
    assert "norm_epsilon" not in exp and exp["layernorm_epsilon"] == CFG["megatron_model"]["norm_epsilon"] == 1e-6


def test_smoke_config_changes_only_steps_and_intervals():
    sc = smoke_config(CFG)
    n = CFG["smoke"]["steps"]
    assert n >= 2 and CFG["run"]["num_steps"] == 100            # the real config is not modified
    real, smoke = build_args(CFG, PATHS), build_args(sc, dict(PATHS, eval_data="E16.jsonl"))
    assert len(real) == len(smoke)
    flag_of = lambda i: next(real[j] for j in range(i, -1, -1) if real[j].startswith("--"))   # noqa: E731
    diff = {flag_of(i) for i in range(len(real)) if real[i] != smoke[i]}
    assert diff == {"--num-rollout", "--eval-interval", "--save-interval", "--eval-prompt-data"}, diff
    assert raises(smoke_config, changed("smoke", steps=1))       # step 2 is what tests the weight push


def test_resume_folder_start_weights_and_plugins_are_wired():
    f = build_args(CFG, PATHS)
    assert value(f, "--load") == value(f, "--save") == "SAVE"   # a resubmitted job resumes
    assert value(f, "--ref-load") == "REF"                       # start weights (slime: --load empty -> ref)
    assert value(f, "--custom-rm-path") == "src.reward.reward"
    assert value(f, "--custom-generate-function-path") == "src.slime_hooks.generate"
    assert value(f, "--data-source-path") == "src.slime_hooks.TextOnlyDataSource"
    assert "--apply-chat-template" not in f


def test_sampling_reward_or_model_differing_from_control_is_refused():
    for key, new in [("temperature", 0.7), ("top_p", 1.0), ("top_k", 50), ("min_p", 0.05),
                     ("presence_penalty", 0.0), ("repetition_penalty", 1.1)]:
        assert raises(check_against_control, changed("sampling", **{key: new}), CONTROL), key
    assert raises(check_against_control, changed("reward", rule="last_answer_field_v2"), CONTROL)
    assert raises(check_against_control, changed("model", revision="main"), CONTROL)
    assert raises(check_against_control, changed("model", chat_template_sha256="0" * 64), CONTROL)


def _config_json(text_config: dict) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(json.dumps({"text_config": text_config}))
    return d / "config.json"


def test_architecture_is_qwen35_4b_and_must_match_config_json():
    check_hf_config(CFG, _config_json(QWEN35_4B))                # the real 4B numbers pass
    assert raises(check_hf_config, CFG, _config_json(dict(QWEN35_4B, hidden_size=2048)))
    moved = {k: v for k, v in QWEN35_4B.items() if k not in ("rope_theta", "partial_rotary_factor")}
    moved["rope_parameters"] = {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
    check_hf_config(CFG, _config_json(moved))                    # transformers 5 layout


def test_real_model_config_json_if_cached():
    snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
                                  + CFG["model"]["revision"] + "/config.json"))
    if not snaps:
        return "SKIP: Qwen3.5-4B config.json not in the local HF cache"
    check_hf_config(CFG, snaps[0])


def test_expected_resolved_values_agree_with_the_flags():
    f = build_args(CFG, PATHS)
    exp = expected_resolved(CFG)
    checked = 0
    for key, want in exp.items():
        flag_key = {v: k for k, v in PARSED_NAME.items()}.get(key, key)
        flag = "--" + flag_key.replace("_", "-")
        if want is None:
            assert flag not in f, key
        elif isinstance(want, bool):
            if flag in f:                       # switches appear only when on
                assert want is True, (key, want)
                checked += 1
        elif flag in f:
            assert value(f, flag) == _flag_value(want), (key, value(f, flag), want)
            checked += 1
    assert checked >= 50, checked
    assert set(HF_CONFIG_KEYS) - {"norm_epsilon"} <= set(exp)


def test_submit_preflight_can_read_its_config_keys_with_sed():
    """submit.sh reads control_config, output_dir, hf_repo and revision with sed on the login node
    (python 3.6 there): each must be ONE quoted line `key = "value"` in the config."""
    text = (REPO_ROOT / "configs" / "train_reinit.toml").read_text()
    for key, want in (("control_config", CFG["run"]["control_config"]), ("output_dir", CFG["data"]["output_dir"]),
                      ("hf_repo", CFG["model"]["hf_repo"]), ("revision", CFG["model"]["revision"])):
        lines = re.findall(rf'(?m)^{key} = "(.*)"$', text)
        assert lines == [want], (key, lines)
    sub = (REPO_ROOT / "scripts" / "deltaai" / "submit.sh").read_text()
    assert 'preflight_slime || { echo "Preflight failed: not submitted."; exit 1; }' in sub
    assert "src.train_args --hf-checkpoint" in sub and "--check" in sub
