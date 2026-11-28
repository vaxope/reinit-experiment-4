"""Turn configs/train_reinit.toml into slime's command-line flags, after checking it.

Purpose : the ONLY place where training parameters become slime flags, so the flags slime
          sees always equal the parameter sheet. Before producing anything it checks:
            1. model, chat template, sampling, answer rule and SGLang kernels equal those of the
               control run the training is compared with ([run] control_config);
            2. the DAPO settings ([algorithm], [overlong]) are consistent: clip-higher upper >= lower,
               dynamic sampling has a filter and over-samples at least one batch, KL coefficient
               only when KL is on, one optimizer update per step;
            3. the GPU layout ([hardware]) is disaggregated: training and generation GPUs add up to
               the node, Megatron's parallel sizes divide the training GPUs and the model's heads,
               no colocation flag can appear in the output, only the synchronous rollout is
               accepted, the micro-batch holds one full sequence, the optimizer offload is paired
               as Megatron requires with fp32 state, and the memory estimate fits the GPU;
            4. the checkpoints of the run fit [checkpoint] disk_budget_gb;
            5. (on DeltaAI) the Megatron architecture numbers equal the model's config.json.
Inputs  : configs/train_reinit.toml, its control config, and paths that differ per user/run
          (HF checkpoint, Megatron checkpoint folders, slime input files) given on the CLI.
Output  : the flag list, one flag or value per line (the Slurm script reads it into an array),
          also written to runs/<run>/slime_args*.txt so every run records what it ran with.
Pipeline: configs/train_reinit.toml --[train_args]--> flags --> `python train.py <flags>`
          (slime, inside the slime container on DeltaAI: slime_setup.sbatch / smoke_train.sh).
Origin  : Reinitialized-Thinking-Experiment-3 @ 97e4706 (plain GRPO, 1 GPU, --colocate); changed
          2026-11-24 for DAPO and separate rollout/training GPUs (vault: decisions/2026-11-22-*).

Usage   : python3 -m src.train_args --hf-checkpoint DIR --ref-load DIR --save DIR \\
              [--config configs/train_reinit.toml] [--out FILE] [--smoke --eval-data FILE]
          python3 -m src.train_args --hf-checkpoint DIR --model-args   (conversion flags only)
          python3 -m src.train_args --hf-checkpoint DIR --check        (all checks, no output;
                                                                        first step of the setup job)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tomllib
from pathlib import Path

from src.config import REPO_ROOT, load_config

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train_reinit.toml"

# Our plug-ins, as import paths slime loads with its load_function (the repo root must be on
# PYTHONPATH inside the Ray workers; train_inside.sh sets it).
DATA_SOURCE = "src.slime_hooks.TextOnlyDataSource"   # keeps our prompt text exactly as built
GENERATE_HOOK = "src.slime_hooks.generate"           # adds min_p / presence / repetition penalty
REWARD_HOOK = "src.reward.reward"                     # rule v3 reward (+ overlong) + per-rollout log

# slime flags build_args never emits; check_flags refuses any list that contains one.
FORBIDDEN_FLAGS = (
    "--colocate",                        # training and generation must stay on separate GPUs
    "--offload", "--offload-train", "--offload-rollout",   # colocation's model offload (the optimizer
                                         # CPU offload, --optimizer-cpu-offload, is a different flag)
    "--use-tis",                         # truncated importance sampling: not part of DAPO
    "--use-score-centering",
    "--apply-chat-template",             # our prompt text must reach the model unchanged
    "--disable-grpo-std-normalization",  # DAPO keeps the group std
)

ROLLOUT_MODES = ("sync",)          # accepted [hardware] rollout_mode values (see check_hardware)
OVERLONG_MODES = ("none", "soft_punish")
WEIGHT_TRANSPORTS = ("nccl", "disk")   # slime --update-weight-transport choices (arguments.py @ 79aa41a)
# slime's filter on the reward value; wrong while the reward has the overlong penalty in it
REWARD_STD_FILTERS = ("slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
                      "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std_with_fallback")
FP32 = "fp32"

# Our [megatron_model] key -> key in the model's config.json (text_config).
HF_CONFIG_KEYS = {
    "num_layers": "num_hidden_layers",
    "hidden_size": "hidden_size",
    "ffn_hidden_size": "intermediate_size",
    "num_attention_heads": "num_attention_heads",
    "num_query_groups": "num_key_value_heads",
    "kv_channels": "head_dim",
    "vocab_size": "vocab_size",
    "rotary_base": "rope_theta",
    "rotary_percent": "partial_rotary_factor",
    "norm_epsilon": "rms_norm_eps",
    "max_position_embeddings": "max_position_embeddings",
}

REQUIRED = {
    "run": ["name", "seed", "num_steps", "control_config"],
    "model": ["hf_repo", "revision", "chat_template_sha256"],
    "megatron_model": ["spec", "flags", "normalization", "position_embedding_type",
                       "attention_dropout", "hidden_dropout", *HF_CONFIG_KEYS],
    "numerics": ["accumulate_allreduce_grads_in_fp32", "attention_softmax_in_fp32",
                 "gradient_accumulation_fusion"],
    "data": ["source_dir", "prompt_mode", "shuffle_salt", "output_dir", "input_key",
             "label_key", "metadata_key"],
    "reward": ["rule", "correct", "incorrect"],
    "overlong": ["mode", "buffer_len", "penalty_factor"],
    "job": ["eval_after_training", "eval_reserve_min", "min_train_min"],
    "smoke": ["steps", "force_max_len_groups", "eval_questions", "serve_questions", "serve_max_new_tokens"],
    "sampling": ["temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty"],
    "rollout": ["n_samples_per_prompt", "rollout_batch_size", "global_batch_size",
                "max_response_len", "max_prompt_len", "shuffle", "shuffle_seed"],
    "algorithm": ["name", "advantage_estimator", "std_normalization", "eps_clip", "eps_clip_high",
                  "per_token_loss", "dynamic_sampling", "dynamic_sampling_filter",
                  "over_sampling_batch_size", "max_consecutive_dropped_groups", "use_kl_loss", "kl_loss_type", "kl_loss_coef",
                  "entropy_coef", "use_tis"],
    "optimizer": ["optimizer", "lr", "lr_decay_style", "lr_warmup_iters", "adam_beta1",
                  "adam_beta2", "adam_eps", "weight_decay", "clip_grad"],
    "eval": ["interval", "n_samples_per_prompt", "max_response_len", "name"],
    "checkpoint": ["save_interval", "est_save_gb", "disk_budget_gb"],
    "hardware": ["rollout_mode", "actor_num_gpus", "rollout_num_gpus", "rollout_num_gpus_per_engine",
                 "gpus_per_node", "tensor_model_parallel_size", "pipeline_model_parallel_size",
                 "context_parallel_size", "sequence_parallel", "use_distributed_optimizer",
                 "optimizer_cpu_offload", "use_precision_aware_optimizer", "optimizer_offload_fraction",
                 "main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype",
                 "update_weight_transport", "sglang_mem_fraction_static",
                 "sglang_attention_backend", "sglang_mm_attention_backend",
                 "max_tokens_per_gpu", "log_probs_chunk_size", "recompute_granularity",
                 "recompute_method", "recompute_num_layers", "gpu_memory_gb", "usable_fraction",
                 "params_total_billion", "params_replicated_billion", "activation_reserve_gb"],
}


def load_train_config(path: str | Path = DEFAULT_TRAIN_CONFIG) -> dict:
    """Parse the training config and verify every required key is present."""
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    missing = [f"{s}.{k}" for s, ks in REQUIRED.items() for k in ks if k not in cfg.get(s, {})]
    if missing:
        raise KeyError(f"{path}: missing config keys: {missing}")
    cfg["_meta"] = {"path": str(path)}
    return cfg


def load_control_for(cfg: dict, override: str | Path | None = None) -> dict:
    """The control config this training is compared with ([run] control_config, or `override`)."""
    path = Path(override) if override else REPO_ROOT / cfg["run"]["control_config"]
    if not path.exists():
        raise FileNotFoundError(
            f"control config {path} not found: training must match the control run it is compared with "
            f"([run] control_config). Write it when that control run pins its model revision and prompts.")
    return load_config(path)


# ---------------------------------------------------------------- checks
def check_against_control(cfg: dict, control: dict) -> None:
    """Training must sample, score and start from exactly what its baseline used."""
    for key in REQUIRED["sampling"]:
        if cfg["sampling"][key] != control["sampling"][key]:
            raise ValueError(f"[sampling] {key} = {cfg['sampling'][key]!r} differs from the control "
                             f"run's {control['sampling'][key]!r}")
    if cfg["reward"]["rule"] != control["scoring"]["extraction_rule"]:
        raise ValueError(f"[reward] rule {cfg['reward']['rule']!r} != control extraction_rule "
                         f"{control['scoring']['extraction_rule']!r}")
    for key in ("revision", "chat_template_sha256"):
        if cfg["model"][key] != control["model"][key]:
            raise ValueError(f"[model] {key} differs from the control config")
    if cfg["model"]["hf_repo"] != control["model"]["name"]:
        raise ValueError(f"[model] hf_repo {cfg['model']['hf_repo']!r} differs from the control config's "
                         f"[model] name {control['model']['name']!r}")
    # the GPQA evaluation runs inside the training job (train_reinit.sbatch), on this node's GPUs
    if control["server"]["dp_size"] > cfg["hardware"]["gpus_per_node"]:
        raise ValueError(f"control [server] dp_size {control['server']['dp_size']} > [hardware] gpus_per_node "
                         f"{cfg['hardware']['gpus_per_node']}: the in-job evaluation could not start")
    # the same attention kernels as the control run / GPQA evaluation (smoke job 3633318)
    for key in ("attention_backend", "mm_attention_backend"):
        if cfg["hardware"]["sglang_" + key] != control["server"][key]:
            raise ValueError(f"[hardware] sglang_{key} = {cfg['hardware']['sglang_' + key]!r} differs from "
                             f"the control config's [server] {key} = {control['server'][key]!r}")


def check_algorithm(cfg: dict) -> None:
    """DAPO settings must be internally consistent (their VALUES are the user's to choose)."""
    al, r, ov = cfg["algorithm"], cfg["rollout"], cfg["overlong"]
    if al["name"] != "dapo":
        raise ValueError("[algorithm] name must be 'dapo' (user, 2026-11-22)")
    if al["advantage_estimator"] != "grpo":
        raise ValueError("[algorithm] advantage_estimator must be 'grpo' (DAPO's group-normalised advantage)")
    if not al["std_normalization"]:
        raise ValueError("[algorithm] std_normalization must be true (DAPO divides by the group std)")
    if not 0 < al["eps_clip"] <= al["eps_clip_high"]:
        raise ValueError("[algorithm] need 0 < eps_clip <= eps_clip_high (clip-higher widens the upper side)")
    if al["dynamic_sampling"]:
        if not al["dynamic_sampling_filter"]:
            raise ValueError("[algorithm] dynamic_sampling needs dynamic_sampling_filter")
        if ov["mode"] != "none" and al["dynamic_sampling_filter"] in REWARD_STD_FILTERS:
            raise ValueError("[algorithm] slime's reward-std filter sees the overlong penalty, so it keeps all-wrong "
                             "groups that differ in length; use src.dapo_filters.check_correctness_varies")
        if al["over_sampling_batch_size"] < r["rollout_batch_size"]:
            raise ValueError("[algorithm] over_sampling_batch_size must be >= [rollout] rollout_batch_size")
        if al["max_consecutive_dropped_groups"] < al["over_sampling_batch_size"]:
            raise ValueError("[algorithm] max_consecutive_dropped_groups must be >= over_sampling_batch_size "
                             "(at least one full sampling round before the run is stopped)")
    if al["use_kl_loss"] and al["kl_loss_coef"] <= 0:
        raise ValueError("[algorithm] use_kl_loss needs kl_loss_coef > 0")
    if not al["use_kl_loss"] and al["kl_loss_coef"] != 0:
        raise ValueError("[algorithm] kl_loss_coef must be 0 when use_kl_loss is false (no hidden KL)")
    if al["use_tis"]:
        raise ValueError("[algorithm] use_tis must be false (not part of DAPO)")
    if al["entropy_coef"] != 0:
        raise ValueError("[algorithm] entropy_coef must be 0 (not part of DAPO)")
    if r["rollout_batch_size"] * r["n_samples_per_prompt"] != r["global_batch_size"]:
        raise ValueError("questions per step x answers per question must equal global_batch_size "
                         "(exactly one optimizer update per step)")
    if cfg["eval"]["max_response_len"] != r["max_response_len"]:
        raise ValueError("validation must use the training response limit")
    if ov["mode"] not in OVERLONG_MODES:
        raise ValueError(f"[overlong] mode must be one of {OVERLONG_MODES}")
    if ov["mode"] == "soft_punish":
        if not 0 < ov["buffer_len"] < r["max_response_len"]:
            raise ValueError("[overlong] need 0 < buffer_len < [rollout] max_response_len")
        if ov["penalty_factor"] <= 0:
            raise ValueError("[overlong] penalty_factor must be > 0")
    if cfg["data"]["prompt_mode"] != "reinit":
        raise ValueError("[data] prompt_mode must be 'reinit' (the experiment trains the reinit format)")


def memory_estimate_gb(cfg: dict) -> dict:
    """Per-training-GPU memory ESTIMATE (formula in configs/train_reinit.toml, [hardware] memory
    section). Worst micro-batch = max(max_tokens_per_gpu, one full prompt + response)."""
    hw, ro, m = cfg["hardware"], cfg["rollout"], cfg["megatron_model"]
    tp = hw["tensor_model_parallel_size"]
    dp = hw["actor_num_gpus"] // (tp * hw["pipeline_model_parallel_size"] * hw["context_parallel_size"])
    params = ((hw["params_total_billion"] - hw["params_replicated_billion"]) / tp
              + hw["params_replicated_billion"]) * 1e9 / hw["pipeline_model_parallel_size"]
    optim = 0 if hw["optimizer_cpu_offload"] else 12 / (dp if hw["use_distributed_optimizer"] else 1)
    state = params * (2 + 4 + optim) / 1e9                       # bf16 weights + fp32 grads (+ optimizer)
    tokens = max(hw["max_tokens_per_gpu"], ro["max_prompt_len"] + ro["max_response_len"]) / hw["context_parallel_size"]
    logits = tokens * m["vocab_size"] / tp * (4 + 4 + 2) / 1e9   # fp32 logits + fp32 grad + bf16 grad
    total = state + logits + hw["activation_reserve_gb"]
    return {"model_state": state, "logits": logits, "activations": hw["activation_reserve_gb"], "total": total,
            "limit": hw["gpu_memory_gb"] * hw["usable_fraction"]}


def disk_estimate_gb(cfg: dict) -> float:
    """Checkpoint saves of the run x est_save_gb: slime saves when (r + 1) % save_interval == 0 and
    after the last rollout (train.py should_run_periodic_action)."""
    n, k = cfg["run"]["num_steps"], cfg["checkpoint"]["save_interval"]
    saves = n // k + (0 if n % k == 0 else 1)
    return saves * cfg["checkpoint"]["est_save_gb"]


def check_hardware(cfg: dict) -> None:
    """Separate training and generation GPUs on one node; Megatron's split must fit the training GPUs
    and the model; the worst micro-batch must fit the memory estimate; saves must fit the disk budget."""
    hw = cfg["hardware"]
    if hw["rollout_mode"] not in ROLLOUT_MODES:
        raise ValueError(f"[hardware] rollout_mode {hw['rollout_mode']!r} not supported (only {ROLLOUT_MODES}): "
                         "slime's fully-async rollout has no evaluation mode at 79aa41a, so checkpoints "
                         "could not be chosen on validation (decision 2026-11-22)")
    if hw["actor_num_gpus"] < 1 or hw["rollout_num_gpus"] < 1:
        raise ValueError("[hardware] training and generation each need at least one GPU of their own")
    if hw["actor_num_gpus"] + hw["rollout_num_gpus"] != hw["gpus_per_node"]:
        raise ValueError("[hardware] actor_num_gpus + rollout_num_gpus must equal gpus_per_node (one node, no sharing)")
    if hw["rollout_num_gpus"] % hw["rollout_num_gpus_per_engine"]:
        raise ValueError("[hardware] rollout_num_gpus must be a multiple of rollout_num_gpus_per_engine")
    split = hw["tensor_model_parallel_size"] * hw["pipeline_model_parallel_size"] * hw["context_parallel_size"]
    if hw["actor_num_gpus"] % split:
        raise ValueError("[hardware] tensor x pipeline x context parallel sizes must divide actor_num_gpus")
    tp, m = hw["tensor_model_parallel_size"], cfg["megatron_model"]
    for key in ("num_attention_heads", "num_query_groups"):
        if m[key] % tp:
            raise ValueError(f"[hardware] tensor_model_parallel_size {tp} must divide [megatron_model] {key} {m[key]}")
    if hw["sequence_parallel"] and tp == 1:
        raise ValueError("[hardware] sequence_parallel needs tensor_model_parallel_size > 1 (Megatron turns it off)")
    ro = cfg["rollout"]
    if hw["max_tokens_per_gpu"] < ro["max_prompt_len"] + ro["max_response_len"]:
        raise ValueError("[hardware] max_tokens_per_gpu must hold one full sequence "
                         "([rollout] max_prompt_len + max_response_len)")
    if hw["optimizer_cpu_offload"] and not hw["use_precision_aware_optimizer"]:
        raise ValueError("[hardware] optimizer_cpu_offload needs use_precision_aware_optimizer (Megatron asserts it)")
    if hw["use_precision_aware_optimizer"] and not hw["use_distributed_optimizer"]:
        raise ValueError("[hardware] use_precision_aware_optimizer needs use_distributed_optimizer (Megatron raises)")
    for key in ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        if hw[key] != FP32:
            raise ValueError(f"[hardware] {key} must be fp32 (master weights, gradients and Adam state stay full precision)")
    if hw["optimizer_cpu_offload"] and hw["optimizer_offload_fraction"] != 1.0:
        raise ValueError("[hardware] optimizer_offload_fraction must be 1.0 (all of the optimizer on the CPU)")
    if hw["update_weight_transport"] not in WEIGHT_TRANSPORTS:
        raise ValueError(f"[hardware] update_weight_transport must be one of {WEIGHT_TRANSPORTS}")
    mem = memory_estimate_gb(cfg)
    if mem["total"] > mem["limit"]:
        raise ValueError(f"[hardware] estimated {mem['total']:.0f} GB per training GPU (model state {mem['model_state']:.0f}"
                         f" + logits {mem['logits']:.0f} + activations {mem['activations']:.0f}) exceeds "
                         f"{mem['limit']:.0f} GB: raise tensor parallel, offload the optimizer or shorten the sequences")
    job = cfg["job"]
    if not isinstance(job["eval_after_training"], bool):
        raise ValueError("[job] eval_after_training must be true or false")
    if job["eval_reserve_min"] <= 0 or job["min_train_min"] <= 0:
        raise ValueError("[job] eval_reserve_min and min_train_min must be positive minutes")
    disk = disk_estimate_gb(cfg)
    if disk > cfg["checkpoint"]["disk_budget_gb"]:
        raise ValueError(f"[checkpoint] about {disk:.0f} GB of saves exceed disk_budget_gb "
                         f"{cfg['checkpoint']['disk_budget_gb']}: save less often")


def check_hf_config(cfg: dict, config_json: Path) -> None:
    """Every Megatron architecture number must equal the model's own config.json."""
    raw = json.loads(Path(config_json).read_text())
    text = raw.get("text_config", raw)          # Qwen3.5 nests the language model config
    rope = text.get("rope_parameters") or {}     # transformers 5 keeps rope values here
    bad = []
    for ours, theirs in HF_CONFIG_KEYS.items():
        got = text.get(theirs, rope.get(theirs))
        if got != cfg["megatron_model"][ours]:
            bad.append(f"{ours}={cfg['megatron_model'][ours]!r} but config.json {theirs}={got!r}")
    if bad:
        raise ValueError("[megatron_model] does not match the model's config.json: " + "; ".join(bad))


def check_flags(flags: list[str]) -> None:
    present = [f for f in flags if f in FORBIDDEN_FLAGS]
    if present:
        raise ValueError(f"forbidden flags in the output (colocation/offload/outside DAPO): {present}")


def check_all(cfg: dict, control: dict) -> None:
    check_against_control(cfg, control)
    check_algorithm(cfg)
    check_hardware(cfg)


def check_data(cfg: dict, out_dir: Path) -> None:
    """The slime input files must be built from the CURRENT prompts and settings (review 2026-11-24).

    src/train_prep.py bakes the prompt text into train.jsonl / val.jsonl. If src/prompts.py changes
    (the final few-shot / reworded reinit prompt) and train_prep is not rerun, training would
    silently use the old prompt while the control uses the new one. Its manifest records the prompt
    hash, mode, salt, length cap and output hashes; any difference, or a half-copied file, stops here.
    """
    from src import prompts   # standard library only; the prompt text lives there
    from src.train_data import sha256_file
    man_path = out_dir / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"{man_path} not found: run `python3 -m src.train_prep` (and copy its output)")
    man = json.loads(man_path.read_text())
    rerun = "rerun `python3 -m src.train_prep --overwrite` and copy the output"
    if man.get("prompt_hash") != prompts.PROMPT_HASH:
        raise ValueError(f"{out_dir} was built with prompt hash {man.get('prompt_hash')!r}, but src/prompts.py "
                         f"is now {prompts.PROMPT_HASH!r}: {rerun}")
    for key in ("prompt_mode", "shuffle_salt"):
        if man.get(key) != cfg["data"][key]:
            raise ValueError(f"{out_dir} was built with {key} {man.get(key)!r}, config says "
                             f"{cfg['data'][key]!r}: {rerun}")
    checks = man.get("checks", {})
    cap = cfg["rollout"]["max_prompt_len"]
    if checks.get("prompt_tokens_max") is None or checks["prompt_tokens_max"] > cap:
        raise ValueError(f"{out_dir}: longest prompt {checks.get('prompt_tokens_max')} tokens > [rollout] "
                         f"max_prompt_len {cap} (slime would drop it silently): {rerun}")
    for split, want in man.get("outputs_sha256", {}).items():
        f = out_dir / f"{split}.jsonl"
        if not f.exists() or sha256_file(f) != want:
            raise ValueError(f"{f} is missing or differs from its manifest (half-copied or edited?): {rerun}")


# ---------------------------------------------------------------- flag building
def _flag_value(v) -> str:
    if isinstance(v, bool):
        raise TypeError("booleans are switches, not values")
    return repr(v) if isinstance(v, float) else str(v)


def model_args(cfg: dict) -> list[str]:
    """The flags that build the Megatron model: the architecture, plus the model-building switches
    that must match our image. Used by build_args AND by the one-time weight conversion
    (tools/convert_hf_to_torch_dist.py), so both always describe and build the same model."""
    m = cfg["megatron_model"]
    a = ["--spec", *m["spec"], *m["flags"]]
    for key in HF_CONFIG_KEYS:
        a += ["--" + key.replace("_", "-"), _flag_value(m[key])]
    a += ["--normalization", m["normalization"], "--position-embedding-type", m["position_embedding_type"],
          "--attention-dropout", _flag_value(m["attention_dropout"]),
          "--hidden-dropout", _flag_value(m["hidden_dropout"])]
    # Megatron's default is on, and building the model then requires APEX's compiled extension,
    # which the image lacks (job 3622008). Off = the same gradients, without the fused kernel.
    if not cfg["numerics"]["gradient_accumulation_fusion"]:
        a.append("--no-gradient-accumulation-fusion")
    return a


def build_args(cfg: dict, paths: dict) -> list[str]:
    """The slime command line, grouped as in slime's own run scripts.

    paths: hf_checkpoint (HF model folder), ref_load (Megatron copy of the start model: the starting
    weights, and the frozen KL reference if KL is on), save (checkpoint folder; also --load, so a
    resubmitted job resumes), prompt_data and eval_data (slime input files from src/train_prep.py).
    """
    d, ro, al, o = cfg["data"], cfg["rollout"], cfg["algorithm"], cfg["optimizer"]
    ev, hw, nu, s = cfg["eval"], cfg["hardware"], cfg["numerics"], cfg["sampling"]
    a: list[str] = []

    def opt(flag, value):
        a.extend([flag, _flag_value(value)])

    # -- hardware: 1 node; training (Megatron) and generation (SGLang) on SEPARATE GPUs. No
    #    --colocate, so slime keeps both resident (no offload) and pushes weights over NCCL.
    a += ["--actor-num-nodes", "1"]
    opt("--actor-num-gpus-per-node", hw["actor_num_gpus"])
    opt("--rollout-num-gpus", hw["rollout_num_gpus"])
    opt("--rollout-num-gpus-per-engine", hw["rollout_num_gpus_per_engine"])
    opt("--num-gpus-per-node", hw["gpus_per_node"])   # slime's default is 8; a GH200 node has 4
    opt("--sglang-mem-fraction-static", hw["sglang_mem_fraction_static"])
    opt("--update-weight-transport", hw["update_weight_transport"])            # trainer -> SGLang weight push
    if hw["update_weight_transport"] == "disk":
        if not paths.get("update_weight_disk_dir"):
            raise ValueError("update_weight_transport 'disk' needs paths['update_weight_disk_dir'] (--update-weight-disk-dir)")
        opt("--update-weight-disk-dir", paths["update_weight_disk_dir"])
    opt("--sglang-attention-backend", hw["sglang_attention_backend"])          # not SGLang's FA3 default
    opt("--sglang-mm-attention-backend", hw["sglang_mm_attention_backend"])

    # -- model architecture (slime scripts/models/qwen3.5-*.sh format)
    a += model_args(cfg)

    # -- checkpoints: start = converted base model (slime loads --ref-load when --load is empty);
    #    save and resume in one folder
    opt("--hf-checkpoint", paths["hf_checkpoint"])
    opt("--ref-load", paths["ref_load"])
    opt("--load", paths["save"])
    opt("--save", paths["save"])
    opt("--save-interval", cfg["checkpoint"]["save_interval"])
    opt("--seed", cfg["run"]["seed"])

    # -- data and our plug-ins
    opt("--prompt-data", paths["prompt_data"])
    opt("--input-key", d["input_key"])
    opt("--label-key", d["label_key"])
    opt("--metadata-key", d["metadata_key"])
    opt("--data-source-path", DATA_SOURCE)
    opt("--custom-generate-function-path", GENERATE_HOOK)
    opt("--custom-rm-path", REWARD_HOOK)
    if ro["shuffle"]:
        a.append("--rollout-shuffle")
    opt("--rollout-seed", ro["shuffle_seed"])

    # -- rollout: questions, answers, lengths, sampling (min_p/penalties added by the hook).
    #    rollout_mode "sync" = slime's standard rollout function, so no --rollout-function-path.
    opt("--num-rollout", cfg["run"]["num_steps"])
    opt("--rollout-batch-size", ro["rollout_batch_size"])
    opt("--n-samples-per-prompt", ro["n_samples_per_prompt"])
    opt("--global-batch-size", ro["global_batch_size"])
    opt("--rollout-max-response-len", ro["max_response_len"])
    opt("--rollout-max-prompt-len", ro["max_prompt_len"])
    opt("--rollout-temperature", s["temperature"])
    opt("--rollout-top-p", s["top_p"])
    opt("--rollout-top-k", s["top_k"])

    # -- validation: same sampling and limit, held-out SuperGPQA slice
    opt("--eval-interval", ev["interval"])
    a += ["--eval-prompt-data", ev["name"], str(paths["eval_data"])]
    opt("--eval-input-key", d["input_key"])
    opt("--eval-label-key", d["label_key"])
    opt("--n-samples-per-eval-prompt", ev["n_samples_per_prompt"])
    opt("--eval-max-response-len", ev["max_response_len"])
    opt("--eval-max-prompt-len", ro["max_prompt_len"])
    opt("--eval-temperature", s["temperature"])
    opt("--eval-top-p", s["top_p"])
    opt("--eval-top-k", s["top_k"])

    # -- DAPO ([algorithm]); overlong shaping is in src/reward.py ([overlong]), not a slime flag
    opt("--advantage-estimator", al["advantage_estimator"])   # std normalisation is slime's default
    opt("--eps-clip", al["eps_clip"])
    opt("--eps-clip-high", al["eps_clip_high"])                # clip-higher
    if al["per_token_loss"]:
        a.append("--calculate-per-token-loss")                 # token-level loss
    if al["dynamic_sampling"]:                                 # drop all-same-reward groups
        opt("--dynamic-sampling-filter-path", al["dynamic_sampling_filter"])
        opt("--over-sampling-batch-size", al["over_sampling_batch_size"])
    if al["use_kl_loss"]:
        a.append("--use-kl-loss")
        opt("--kl-loss-type", al["kl_loss_type"])
        opt("--kl-loss-coef", al["kl_loss_coef"])
    opt("--entropy-coef", al["entropy_coef"])

    # -- optimizer (AdamW; every value explicit)
    for key in ("optimizer", "lr", "lr_decay_style", "lr_warmup_iters", "adam_beta1",
                "adam_beta2", "adam_eps", "weight_decay", "clip_grad"):
        opt("--" + key.replace("_", "-"), o[key])

    # -- memory / parallelism on the training GPUs (data parallel = actor GPUs / the split below)
    opt("--tensor-model-parallel-size", hw["tensor_model_parallel_size"])
    opt("--pipeline-model-parallel-size", hw["pipeline_model_parallel_size"])
    opt("--context-parallel-size", hw["context_parallel_size"])
    if hw["sequence_parallel"]:
        a.append("--sequence-parallel")
    if hw["use_distributed_optimizer"]:
        a.append("--use-distributed-optimizer")
    if hw["use_precision_aware_optimizer"]:
        a.append("--use-precision-aware-optimizer")
    for key in ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        opt("--" + key.replace("_", "-"), hw[key])               # fp32 (checked), written out explicitly
    if hw["optimizer_cpu_offload"]:
        a.append("--optimizer-cpu-offload")
        opt("--optimizer-offload-fraction", hw["optimizer_offload_fraction"])
    a.append("--use-dynamic-batch-size")
    opt("--max-tokens-per-gpu", hw["max_tokens_per_gpu"])
    opt("--log-probs-chunk-size", hw["log_probs_chunk_size"])
    opt("--recompute-granularity", hw["recompute_granularity"])
    opt("--recompute-method", hw["recompute_method"])
    opt("--recompute-num-layers", hw["recompute_num_layers"])

    # -- numerics
    if nu["accumulate_allreduce_grads_in_fp32"]:
        a.append("--accumulate-allreduce-grads-in-fp32")
    if nu["attention_softmax_in_fp32"]:
        a.append("--attention-softmax-in-fp32")

    check_flags(a)
    return a


# Flags whose value Megatron stores under a different attribute name than the flag spells.
# --norm-epsilon -> args.layernorm_epsilon (Megatron ae76c60e; setup job 3631176).
PARSED_NAME = {"norm_epsilon": "layernorm_epsilon"}


def smoke_config(cfg: dict) -> dict:
    """The config of the smoke run (scripts/deltaai/smoke_train.sh): the real one with ONLY the step
    count changed to [smoke] steps, and validation + checkpoint saving at that step. Everything
    else (batches, lengths, sampling, DAPO, optimizer, GPU layout, memory) is the real run's."""
    c = copy.deepcopy(cfg)
    n = cfg["smoke"]["steps"]
    if n < 2:
        raise ValueError("[smoke] steps must be >= 2: step 2 is what tests the weight push to SGLang")
    c["run"]["num_steps"] = n
    c["eval"]["interval"] = n
    c["checkpoint"]["save_interval"] = n
    return c


def expected_resolved(cfg: dict) -> dict:
    """What slime's parsed `args` must hold after it reads build_args' flags: attribute -> value.

    Checked inside the image by scripts/deltaai/check_slime_args.py (setup job). It catches what
    a flag list alone cannot: slime parses Megatron's flags with ignore_unknown_args=True, so a
    flag it does not know is dropped silently, and its validation fills in values of its own.
    Every value comes from the config; nothing new is decided here.
    """
    m, ro, al, o = cfg["megatron_model"], cfg["rollout"], cfg["algorithm"], cfg["optimizer"]
    ev, hw, nu, s = cfg["eval"], cfg["hardware"], cfg["numerics"], cfg["sampling"]
    exp = {
        # DAPO
        "advantage_estimator": al["advantage_estimator"], "grpo_std_normalization": al["std_normalization"],
        "eps_clip": al["eps_clip"], "eps_clip_high": al["eps_clip_high"],
        "calculate_per_token_loss": al["per_token_loss"],
        "dynamic_sampling_filter_path": al["dynamic_sampling_filter"] if al["dynamic_sampling"] else None,
        "use_kl_loss": al["use_kl_loss"], "kl_loss_type": al["kl_loss_type"], "kl_loss_coef": al["kl_loss_coef"],
        "entropy_coef": al["entropy_coef"], "use_tis": al["use_tis"],
        # GPU layout: separate GPUs, nothing offloaded (slime resolves offload_* to False)
        "colocate": False, "offload_train": False, "offload_rollout": False,
        "actor_num_nodes": 1, "actor_num_gpus_per_node": hw["actor_num_gpus"],
        "rollout_num_gpus": hw["rollout_num_gpus"], "rollout_num_gpus_per_engine": hw["rollout_num_gpus_per_engine"],
        "num_gpus_per_node": hw["gpus_per_node"],
        "tensor_model_parallel_size": hw["tensor_model_parallel_size"],
        "pipeline_model_parallel_size": hw["pipeline_model_parallel_size"],
        "context_parallel_size": hw["context_parallel_size"],
        "use_distributed_optimizer": hw["use_distributed_optimizer"],
        "sequence_parallel": hw["sequence_parallel"],
        "use_precision_aware_optimizer": hw["use_precision_aware_optimizer"],
        "optimizer_cpu_offload": hw["optimizer_cpu_offload"],
        # Megatron turns these strings into torch dtypes (training/arguments.py map_dtype);
        # check_slime_args.py compares torch.float32 with "fp32" by name
        "main_params_dtype": hw["main_params_dtype"], "main_grads_dtype": hw["main_grads_dtype"],
        "exp_avg_dtype": hw["exp_avg_dtype"], "exp_avg_sq_dtype": hw["exp_avg_sq_dtype"],
        "update_weight_transport": hw["update_weight_transport"],
        # batches and lengths
        "n_samples_per_prompt": ro["n_samples_per_prompt"], "rollout_batch_size": ro["rollout_batch_size"],
        "global_batch_size": ro["global_batch_size"], "rollout_max_response_len": ro["max_response_len"],
        "rollout_max_prompt_len": ro["max_prompt_len"], "eval_max_response_len": ev["max_response_len"],
        "rollout_shuffle": ro["shuffle"], "rollout_seed": ro["shuffle_seed"],
        "num_rollout": cfg["run"]["num_steps"], "seed": cfg["run"]["seed"],
        "save_interval": cfg["checkpoint"]["save_interval"], "eval_interval": ev["interval"],
        # sampling slime itself sends (the hook adds min_p and the penalties)
        "rollout_temperature": s["temperature"], "rollout_top_p": s["top_p"], "rollout_top_k": s["top_k"],
        # memory
        "max_tokens_per_gpu": hw["max_tokens_per_gpu"], "log_probs_chunk_size": hw["log_probs_chunk_size"],
        "sglang_mem_fraction_static": hw["sglang_mem_fraction_static"],
        "sglang_attention_backend": hw["sglang_attention_backend"],
        "sglang_mm_attention_backend": hw["sglang_mm_attention_backend"],
        "accumulate_allreduce_grads_in_fp32": nu["accumulate_allreduce_grads_in_fp32"],
        "attention_softmax_in_fp32": nu["attention_softmax_in_fp32"],
        "gradient_accumulation_fusion": nu["gradient_accumulation_fusion"],
    }
    if al["dynamic_sampling"]:
        exp["over_sampling_batch_size"] = al["over_sampling_batch_size"]
    if hw["optimizer_cpu_offload"]:
        exp["optimizer_offload_fraction"] = hw["optimizer_offload_fraction"]
    exp.update({k: o[k] for k in ("optimizer", "lr", "lr_decay_style", "lr_warmup_iters", "adam_beta1",
                                   "adam_beta2", "adam_eps", "weight_decay", "clip_grad")})
    exp.update({k: m[k] for k in ("num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads",
                                   "num_query_groups", "kv_channels", "vocab_size", "rotary_base",
                                   "rotary_percent", "norm_epsilon", "max_position_embeddings",
                                   "attention_dropout", "hidden_dropout")})
    for flag_name, attr in PARSED_NAME.items():   # check the attribute Megatron really sets
        exp[attr] = exp.pop(flag_name)
    return exp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG))
    ap.add_argument("--control-config", default=None, help="default: [run] control_config")
    ap.add_argument("--hf-checkpoint", required=True, help="HF model folder (snapshot @ revision)")
    ap.add_argument("--ref-load", help="Megatron copy of the start model")
    ap.add_argument("--save", help="checkpoint folder (also used to resume)")
    ap.add_argument("--out", default=None, help="also write the flags here (one per line)")
    ap.add_argument("--smoke", action="store_true",
                    help="flags for the smoke run: [smoke] steps, validation and saving at that step")
    ap.add_argument("--eval-data", default=None,
                    help="validation file instead of <output_dir>/val.jsonl (the smoke run's short slice)")
    ap.add_argument("--model-args", action="store_true",
                    help="print only the architecture flags (for the weight conversion)")
    ap.add_argument("--update-weight-disk-dir", default=None,
                    help="folder for the weight push when [hardware] update_weight_transport = 'disk'")
    ap.add_argument("--check", action="store_true",
                    help="run every check (control match, DAPO, GPU layout, config.json) and exit")
    args = ap.parse_args(argv)

    cfg = load_train_config(args.config)
    if args.model_args:   # conversion: architecture checked against config.json, nothing else needed
        check_hf_config(cfg, Path(args.hf_checkpoint) / "config.json")
        sys.stdout.write("\n".join(model_args(cfg)) + "\n")
        return
    if not (args.check or (args.ref_load and args.save)):
        ap.error("--ref-load and --save are required unless --model-args or --check")
    try:
        control = load_control_for(cfg, args.control_config)
    except FileNotFoundError as e:
        sys.exit(f"[train_args] {e}")
    check_all(cfg, control)
    check_hf_config(cfg, Path(args.hf_checkpoint) / "config.json")
    check_data(cfg, REPO_ROOT / cfg["data"]["output_dir"])
    mem = memory_estimate_gb(cfg)
    print(f"[train_args] memory estimate per training GPU: {mem['total']:.0f} GB of {mem['limit']:.0f} "
          f"(model state {mem['model_state']:.0f}, logits {mem['logits']:.0f}, activations {mem['activations']:.0f}); "
          f"checkpoints ~{disk_estimate_gb(cfg):.0f} GB of {cfg['checkpoint']['disk_budget_gb']}", file=sys.stderr)
    if args.check:
        print(f"[train_args] config {cfg['_meta']['path']} passes every check "
              f"(control {cfg['run']['control_config']})", file=sys.stderr)
        return
    if args.smoke:   # after the checks above, which ran on the real config
        cfg = smoke_config(cfg)
    out_dir = REPO_ROOT / cfg["data"]["output_dir"]
    paths = {"hf_checkpoint": args.hf_checkpoint, "ref_load": args.ref_load, "save": args.save,
             "prompt_data": out_dir / "train.jsonl",
             "eval_data": Path(args.eval_data) if args.eval_data else out_dir / "val.jsonl",
             "update_weight_disk_dir": args.update_weight_disk_dir}
    for key in ("prompt_data", "eval_data"):
        if not Path(paths[key]).exists():
            sys.exit(f"{paths[key]} not found: run `python3 -m src.train_prep` first")
    flags = build_args(cfg, paths)
    print(f"[train_args] {len(flags)} flags for run {cfg['run']['name']!r}", file=sys.stderr)
    text = "\n".join(flags) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
