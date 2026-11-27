# CLAUDE.md: GAR reinitialized-thinking rebuild (control + training)

## Current goal (keep scope to this)

**1. Control outputs (the baseline):** an untrained Qwen model answering **GPQA Diamond** in three modes, **`think`**, **`default`** and **`reinit`**, generated with **SGLang on NCSA DeltaAI**, with every input, output token and error logged. These outputs are the baseline every trained model is compared against, so they must be exactly reproducible and fully transparent.

**Next control run (2026-11-19):** the same control experiment on **Qwen3.5-4B** (the existing code and the completed run `control_v1` use Qwen3.5-2B), with prompts that add a few worked examples so the model writes the answer exactly as asked, and a reworded `reinit` prompt so that `</analysis>` closes before the answer. The prompts are designed together with the training side (training uses the same `reinit` prompt) and frozen before any run. New prompts or a new model mean new run names (e.g. `control_4b_v1`); `control_v1` stays as it is.

**2. Training setup (added 2026-11-22, user's choice; branch `model-training-setup`):** get everything ready to train in the `reinit` format **as soon as the new prompts and the new control run are final**, but **do not train yet**. Training is built in this repository, starting from the slime pipeline of `Reinitialized-Thinking-Experiment-3` (vaxope/Reinitialized-Thinking-Experiment-3; local copy path in `CLAUDE.local.md`), ported one component at a time (Rule 1) and changed in three ways the user chose on 2026-11-22:
- **DAPO instead of plain GRPO** (exact values decided later; see *Training specification*). This replaces Experiment-3's "plain GRPO only" rule.
- **Synchronous disaggregated training instead of colocation:** rollout (SGLang) and training (Megatron) on separate GPUs (2 + 2 on one GH200 node), taking turns each step, no `--colocate`, no CPU offloading, no dependence on `torch_memory_saver` (the colocation path that caused trouble on the ARM GH200 nodes).
- **A code-server on DeltaAI** inside a long-held Slurm allocation, so smoke tests and training attempts run inside one allocation instead of waiting in the queue for each attempt.

The plan and its open questions are in the vault: `decisions/2026-11-22-training-setup-plan.md`. **2026-11-24 (user's request):** everything the 2-step smoke test needs is ported and tested locally (see *Training components*), for Qwen3.5-4B with DAPO and 2 + 2 GPUs; DAPO values and prompts are placeholders that only the config sets. The long training job and the trained-model evaluation job are not ported yet. Nothing has run on DeltaAI. Training runs only after the user says the prompts and control are final.

**Out of scope:** datasets other than GPQA (evaluation only) and SuperGPQA (training); training any mode other than `reinit`.

**Where things run:**
- **DeltaAI:** all model generation (SGLang) and all training (slime), through `scripts/deltaai/submit.sh` only (the code-server job is also submitted through it).
- **Local machine:** writing code, unit tests, data prep, analysis of saved outputs, the trajectory viewer.

## Components

```
pinned GPQA CSV --[src/data.py]--> data/gpqa_diamond.jsonl --[src/prompts.py]--> exact input text
                     [src/generate.py <-> SGLang] in a Slurm job on DeltaAI (scripts/deltaai/)
                                                    |
runs/<run>/generations.jsonl --[src/score.py + src/extract.py]--> scored.jsonl, summary.json, summary.md
                                                    |
            viewer/index.html (offline dashboard) · src/compare_runs.py · src/compare_trajectories.py
```

| File | Role |
|---|---|
| `configs/control.toml` + `src/config.py` | every parameter, with meaning/source/reason; loader checks required keys |
| `src/prompts.py` | builds the exact input per mode (think, default, reinit); deterministic option shuffle |
| `src/extract.py` | answer extraction, rule `last_answer_field_v3` (strict field or answer line), plus the lenient reading |
| `src/data.py` | pinned, hash-verified GPQA CSV -> `data/gpqa_diamond.jsonl`; `--show N` prints prompts |
| `src/schema.py` | field list of a generation record (the generate -> score contract) |
| `src/generate.py` | SGLang client: sends token ids to `/generate`, logs every generation, resumable, SIGTERM-safe. (Its `--checkpoint` option serves a trained model and needs `src/check_trained.py`, still to be ported from Experiment-3; see *Training components (planned)*.) |
| `src/check_run.py` | mechanical checks of a run (prompts rebuilt, token ids, counts, completeness); PASS/FAIL |
| `src/score.py` | scoring + all metrics (v3 accuracy, lenient accuracy, strict v2); never edits the raw log |
| `src/compare_runs.py` | two runs side by side per mode: accuracy with clustered CI, lenient accuracy, truncation, token statistics |
| `src/compare_trajectories.py` | trajectory comparison per mode: answer-line shapes, commitments, invented sources, lengths, fixed/broken question IDs (counts and IDs only) |
| `viewer/index.html` | offline dashboard: tiles, filters, qid search, full trajectory, mode compare |
| `tests/` | `python3 tests/run_all.py` (no pytest needed); `tests/fake_run.py` makes a synthetic run with no GPQA text; `tests/fake_sglang.py` stands in for SGLang |
| `scripts/deltaai/` | `setup.sh` (login node, once), `submit.sh` (the only way to submit: smoke / pilot / full), `smoke_test.sbatch`, `run_control.sbatch`, `serve_and_generate.sh`, `common.sh`, `env.example.sh` (copy to gitignored `env.sh`) |

Everything uses the standard library only (Python 3.12), except the template-parity test, which needs `transformers` (and `jinja2`) in `.venv/`. Run the full suite with `.venv/bin/python tests/run_all.py`.

### Training components

Source: `Reinitialized-Thinking-Experiment-3` @ `97e4706`. **Ported 2026-11-24** (everything the smoke test `submit.sh slime-setup` runs): all rows below except the two marked *not ported*. "Change" is what differs from Experiment-3. `[run] control_config` names `configs/control_4b.toml`, which `train_args.py --check` (step 0 of the setup job and the login-node preflight) compares against. **2026-11-24:** that file exists as a **draft** (= `configs/control.toml` with only `[model]` set to the 4B; revision still a placeholder) so the smoke test can run before the final prompts. The 4B training data is built locally (`data/train/supergpqa_pcb_v1/`: train/val byte-identical to Experiment-3's; `slime_reinit_4b/` with 5,733/5,733 prompts matching the 4B template; GPQA parity on the 4B 2,376/2,376). `train_args.check_data` refuses slime input files built from a different prompt (manifest `prompt_hash` vs `src/prompts.py`), salt or mode, prompts longer than `[rollout] max_prompt_len`, or files that differ from their manifest: **after the final prompts, rerun `src.train_prep --overwrite` and copy the output again.**

```
SuperGPQA (pinned) --[src/train_data.py]--> data/train/<build>/{train,val}.jsonl --[src/train_prep.py]--> slime input (reinit prompt)
configs/train_reinit.toml --[src/train_args.py]--> slime flags (DAPO, disaggregated GPUs)
slime train.py on DeltaAI: SGLang rollout GPU(s) <-> Megatron training GPU(s); src/reward.py, src/slime_hooks.py
     --> checkpoints --[convert + src/check_trained.py]--> HF model --[src/generate.py --checkpoint, the matching control config unchanged]--> scored like control
```

| File | Role | Change from Experiment-3 |
|---|---|---|
| `configs/train_data.toml` + `src/train_data.py` | SuperGPQA physics/chemistry/biology -> dedup -> GPQA decontamination -> 4 options -> train/val | none |
| `configs/train_reinit.toml` + `src/train_args.py` | parameter sheet -> slime flags; refuses sampling/reward differing from control | `[algorithm]` (DAPO flags, consistency checks instead of the plain-GRPO guard), `[hardware]` (2 + 2 GPUs, no `--colocate`/offload, `rollout_mode` only `sync`), Qwen3.5-4B architecture, `[run] control_config`; `--check` mode |
| `src/train_prep.py` | questions -> slime input lines with the exact `build_text` reinit prompt, parity-checked | none (it picks up whatever reinit prompt `src/prompts.py` holds) |
| `src/reward.py` | reward = rule v3 (1/0), every rollout logged | + DAPO overlong penalty (`[overlong]`, `soft_punish` or `none`); log keeps `correctness_reward` and `overlong_penalty` |
| `src/slime_hooks.py` | adds min_p / presence / repetition penalty to slime's sampling; plain-text prompts | smoke only: forces full-length answers when `$GAR_SMOKE_FORCE_MAX_LEN_GROUPS` is set |
| `configs/slime_container.toml` + `src/slime_container.py` + `containers/slime_gh200.def` | arm64/GH200 slime image (image `3aff7962…` built in Experiment-3, job 3620006) | none: the built image is reused (its `torch_memory_saver` is simply unused) |
| `src/check_roundtrip.py` | HF <-> Megatron weight check | none |
| `src/pick_checkpoint.py`, `src/check_trained.py` | pick a checkpoint on SuperGPQA validation (never GPQA), convert, check | none (synchronous training keeps slime's in-loop validation) |
| `scripts/deltaai/eval_trained.sbatch`, `train_reinit.sbatch`, `setup_and_train.sbatch` | the long training job and the trained-model evaluation | *not ported* (after the smoke test) |
| `src/check_smoke.py` | the smoke run's rollout log: steps, weight push, validation, rewards | dynamic sampling (>= 16 complete groups with right AND wrong answers per step, more answers allowed); rewards = correctness + overlong penalty; a full-length answer in a trained step-1 group |
| `src/dapo_filters.py` (new) | DAPO dynamic-sampling filter on correctness (`sample.gar_correct`, set by `reward.py`), plus a `_with_fallback` variant; stops the run after `[algorithm] max_consecutive_dropped_groups` (160) groups in a row without a signal, because slime samples with no round limit | new 2026-11-24; stop rule 2026-11-27 |
| `src/train_report.py` | per-step training summary from the rollout log | none yet (DAPO metrics later) |
| `scripts/deltaai/` (training) | `build_slime.sbatch`, `slime_setup.sbatch` (step 0 = `train_args --check`), `train_inside.sh`, `slime_common.sh`, `smoke_train.sh`, `check_slime_args.py`, `check_slime_gpu.py`, `convert_to_hf.py`; `submit.sh build-slime / slime-setup` | 4 GPUs (`#SBATCH --gpus-per-node=4`, tested against the config), 64 CPUs / 240G (unverified against node limits), 3 h default; submit.sh shows GPUs × hours and runs the preflight; `setup.sh` also downloads the training model; GPU check 2b reads the attention shapes from the config; `common.sh` / `serve_and_generate.sh` take an optional `CONTROL_CONFIG` (default `configs/control.toml`, unchanged) so the smoke serving test uses the 4B control |
| `scripts/deltaai/slime_env.sh` (new) | wraps every slime-image command (`S()`, the build job's GPU check): `CUDNN_HOME` and `LD_LIBRARY_PATH` = PyTorch's own cuDNN 9.20. TransformerEngine otherwise loads the image's system cuDNN 9.14 first, and the mix gave wrong attention gradients at head size 256 (setup job 3680134; diagnosis jobs 3686595/3686686). `check_slime_gpu.py` check `cudnn` fails on any mix | new 2026-11-27 |
| `scripts/deltaai/code_server.sbatch` (new) | starts a pinned code-server on a compute node inside a long allocation; reached through an SSH tunnel | *not written* (set aside by the user, 2026-11-24) |

---

## Rule 1: Explain everything (the user must understand and control the codebase)

**Every response that proposes, writes or changes code, configs or scripts must explain:**
1. **What** is being done, in plain terms.
2. **How it connects** to the other pieces: what feeds into it, what consumes its output, and its place in the component diagram above.
3. **Why**: the rationale for each choice, the alternatives considered, and what was rejected and why.
4. **How to verify** it works: the test, check or command the user can run.

**Process:**
- **Plan before code.** For any new component, weight out different options, explain the plan (points 1–4) and wait for the user's approval.
- **One component at a time**, each with comments on non-obvious lines, unit tests, and a short note (see Notes).
- **Every file starts with a docstring or header comment** stating its purpose, its inputs and outputs, and where it sits in the pipeline.
- **No magic numbers.** Every parameter (sampling settings, token limits, sample counts, seeds) lives in `configs/`, with a comment giving its meaning, value, source and reason.
- **Walkthroughs on request:** explain any file line by line, or quiz the user on it.
- Never make changes the user didn't ask for. If you notice something worth changing, describe it and ask.

## Rule 2: Git (strict)

- **Never `git commit` or `git push` without explicit permission from the user for that specific commit or push.** Permission for one commit does not carry over to the next. Never force-push, rebase, merge or reset without explicit permission.
- Before asking, show: the files to be committed, a summary of the diff, and the proposed commit message.
- **Do not credit Claude in any form.** No `Co-Authored-By` trailer, no "Generated with Claude Code" line, no Claude attribution in commit messages or PR descriptions. Write commit messages as the user would. (`.claude/settings.json` also turns off the automatic attribution.)

## Rule 3: Consult before acting

**Always stop and ask before:**
- any DeltaAI job (`sbatch`, `salloc`, `srun`), including starting a code-server session and launching training or smoke runs inside one. State the GPUs, the time requested and the estimated GPU-hours (a held allocation is charged while idle);
- installing packages, pulling or building containers, or changing the environment;
- deleting or overwriting data, outputs or logs;
- changing prompts, chat-template handling, sampling settings, answer extraction or metrics once a run has started. These break comparability; a change means a new run name.

When asking, give: **what**, **why**, **files affected**, **cost/risk**, **how to undo**. Reading files, running unit tests and CPU-only checks need no approval.

## Rule 4: Transparency

- **Never let SGLang apply the chat template.** Build each input with `src/prompts.py` and send it as raw text (or token IDs) to SGLang's native `/generate` endpoint. Log both the text and its token IDs.
- **Log every generation** in the format below, **before** any scoring. Errors and exceptions go in their own field and are never dropped. A failed or truncated sample is recorded and counted, not skipped.
- Report only numbers that come from logs. Mark anything unverified.

## Rule 5: Data

- `data/` and `runs/` are gitignored. **GPQA is test-only**: never train on it, never commit or publish its text. Notes and shared pages use question IDs only.

---

## Modes (built by `src/prompts.py`)

Qwen3.5-2B's chat template **turns thinking off by default**: thinking is on only when `enable_thinking=True`. (Qwen3.5-4B's template @ `851bf6e8` is the reverse: on unless `enable_thinking=False`; sha256 `a4aee8af…`. `src/prompts.py` always passes `enable_thinking` explicitly, so the built text is the same; the 4B parity check is still to be run.) `src/prompts.py` builds every input by hand, matching the template (`chat_template.jinja` @ `15852e8c`) exactly; all prompt text lives there.

| Mode | System turn | Assistant turn starts with | What the model writes |
|---|---|---|---|
| `think` | none | `<think>\n` (open) | hidden reasoning, `</think>`, then the answer |
| `default` | none | `<think>\n\n</think>\n\n` (closed, empty) | the answer, usually with visible reasoning |
| `reinit` | persona (`REINIT_SYSTEM_PROMPT`) | `<think>\n\n</think>\n\n<analysis>\n` (thinking off + our open tag) | analysis, `</analysis>`, then the answer |

`think` and `default` use the identical user prompt: a one-line task statement, the question, options `A)`–`D)`, then Qwen's recommended multiple-choice instruction, verbatim from the model card: ``Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.``

`reinit` ("reinitialized thinking", **user's choice 2026-11-15**: full prompt, persona in the system turn, name `reinit`): the user's drafted prompt with agreed fixes. Its user turn asks for a 3-step analysis (deconstruct, multi-step derivation, verification) inside `<analysis></analysis>` and the answer after `</analysis>`, then the same question block and the same answer instruction. `<analysis>` is plain text (several ordinary tokens), not a special token. Parity for `reinit` = official template with thinking off + `"<analysis>\n"`. It differs from `default` in three ways at once (persona, steps, block), so it measures the whole prompt package, not the block alone.

Options are shuffled with seed string `<shuffle_salt>|<qid>|<sample_idx>`, shared by all modes.

## Control-run specification (values live in `configs/control.toml`)

- Model: `Qwen/Qwen3.5-2B` in bf16, at a pinned Hugging Face revision.
- Model revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`. Data: `Idavidrein/gpqa` @ `633f5ee8`, CSV SHA-256 checked.
- Engine: SGLang server on one DeltaAI GPU, attention backend pinned to `flashinfer` and the vision encoder's to `triton_attn` (`[server] attention_backend`, `mm_attention_backend`; container commands run with `CC=gcc,CXX=g++`; FA3 is missing from the arm64 image, smoke job 3611059; generate.py refuses pilot/control runs on another backend); native `/generate`; input sent as token IDs from the pinned tokenizer (text logged too). `generate.py` refuses to run if the tokenizer's chat-template hash differs from `[model] chat_template_sha256` or template parity fails.
- Sampling (**confirmed by the user 2026-11-13**): one setting for all modes, fixed for all future comparisons: Qwen's thinking-mode text-task values, temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 1.5, repetition penalty 1.0. Default mode deliberately does not use Qwen's separate non-thinking values.
- Token limit (**confirmed by the user 2026-11-13**): 81,920 new tokens for all modes (model card value for complex benchmarks), so thinking is effectively unrestricted; the truncation rate is reported.
- Samples per question: 4 (pilot_v1, job 3611388, 2026-11-16: 60/60 clean, about 0.10 SU; full run projected at 1.5–2.5 GPU-hours).
- Answer extraction: `src/extract.extract_answer`, rule `last_answer_field_v3` (**user's choice 2026-11-16**, after pilot_v1 showed only 5/60 strict fields): the last answer field in the visible text (after the first `</think>` in think mode, after the first `</analysis>` in reinit mode; a block that never closes = no answer). A field is the strict `"answer": "X"` or an answer line (`answer` key in any case/quoting, `:`, one capital letter, nothing else on the line; e.g. `answer: C`, `answer": "C"`). Prose ("the answer is C"), bare letters and other keys are not accepted (counted as near misses). Strict-v2 accuracy is reported beside it. **Lenient reading beside v3** (user's choice 2026-11-16, after control_v1): `extract_lenient`, rule `any_answer_lenient_v1` (`[scoring] lenient_rule`): "if it gives an answer, count it": the last answer of any shape anywhere in the output (fields, `<answer>X</answer>`, choice lines, "the answer is X" unless hypothetical, a bare final letter, a final option line); truncated still wrong. Reported as `lenient_answer`/`lenient_correct` and "accuracy, lenient"; never the official accuracy and never the reward.
- Scoring: a truncated generation (`finish_reason` `length`) counts as wrong; accuracy uses all records as the denominator; 95% CI is Wilson (1 sample/question) or clustered by question (>1).

## Training specification (planned; values will live in `configs/train_reinit.toml`)

Decided by the user on 2026-11-22. Everything marked *open* is decided later, before the first training job; once a run starts, any change means a new run name.

- **Kept from Experiment-3 (unless the user changes them):** training data SuperGPQA (`m-a-p/SuperGPQA`, Science / Physics, Chemistry, Biology, 4 options, decontaminated against GPQA); prompt = the `reinit` prompt exactly as `src/prompts.py` builds it; reward from `extract.py` rule v3 (1 correct, 0 otherwise; truncated or `<analysis>` never closed = 0); rollout sampling identical to the control sampling; GPQA never used for training or checkpoint choice; every trained model is evaluated exactly like the control (198 × 4 × three modes, control sampling, 81,920 cap, v3 + lenient).
- **Algorithm: DAPO** (Yu et al. 2025) instead of plain GRPO. Its four parts and how slime @ `79aa41a` provides them (read from slime's `arguments.py` on 2026-11-22):
  - clip-higher: `--eps-clip` (low) and `--eps-clip-high` (high); paper values 0.2 / 0.28;
  - dynamic sampling: `--dynamic-sampling-filter-path` with `--over-sampling-batch-size` (drops groups whose answers are all right or all wrong). Filter = **ours, on correctness** (`src/dapo_filters.check_correctness_varies`; user, 2026-11-24): slime's `check_reward_nonzero_std` looks at the reward, which includes the overlong penalty, and would keep all-wrong groups that differ only in length. `train_args` refuses slime's reward filter while overlong shaping is on;
  - token-level loss: `--calculate-per-token-loss`;
  - overlong reward shaping: **no slime flag**; would be added in `src/reward.py` (soft length penalty in a buffer before the limit) or replaced by overlong filtering (truncated answers masked from the loss);
  - KL: DAPO drops the KL term (`--use-kl-loss` off). *Open:* all values, and whether each part is used.
- **Layout (user, 2026-11-22): synchronous disaggregated** on one full GH200 node: **2 rollout GPUs (SGLang) + 2 training GPUs (Megatron)**, no `--colocate`, standard rollout function (`--actor-num-gpus-per-node 2`, `--rollout-num-gpus 2`). **Training GPUs (2026-11-24, after the review):** tensor parallel 2 + sequence parallel (user's choice; slime's fp32 logits of a 36,864-token micro-batch would need ~69 GB on one GPU at tensor parallel 1), plus optimizer CPU offload with the precision-aware optimizer and fp32 master/grads/Adam (slime's own Qwen3.5 scripts do the same; frees ~31 GB per GPU). `train_args.memory_estimate_gb` refuses configs that would not fit (estimate ~73 of 86 GB usable). NCCL/gloo kept on loopback (`train_inside.sh`); weight push `nccl`, `disk` as a config fallback. Fully on-policy; slime's in-loop validation (`--eval-interval`) keeps working, so checkpoints are still chosen on SuperGPQA validation. Fully async (`slime.rollout.fully_async_rollout`) was rejected: answers up to a step stale and no evaluation mode at `79aa41a`. Cost: about 4 GPU-hours per wall-clock hour.
- **Checkpoints (2026-11-24):** save and validate every 25 steps (was 10): 4 saves × ~60 GB = ~240 GB of a 400 GB budget under the 500 GB quota; `train_args` refuses configs whose saves exceed `[checkpoint] disk_budget_gb`.
- **Smoke test realism (2026-11-24):** step 1 forces the first answer of each of the first 32 groups to the full 32,768 tokens (SGLang `ignore_eos`, smoke only), and `check_smoke` requires such an answer in a trained group, so worst-case memory is tested in the smoke run. `submit.sh slime-setup` runs a login-node preflight (env, control config, data, 4B snapshot, slime image, `train_args --check`) before `sbatch`.
- **Model (user, 2026-11-22): `Qwen/Qwen3.5-4B`**, to match the next control run (`control_4b_v1`); revision pinned when that control run pins it. Architecture flags from slime's `scripts/models/qwen3.5-4B.sh`, checked against the model's `config.json` by `train_args.py`.
- **Code-server (user, 2026-11-22):** first check whether DeltaAI's Open OnDemand offers a code-server / VS Code app (no install). If not: a pinned arm64 code-server release started by `scripts/deltaai/code_server.sbatch` (through `submit.sh`) on a `ghx4` compute node, password from a gitignored file, bound to the node's loopback and reached through an SSH tunnel via the login node. It saves queue waits *between attempts*, not the first wait; GPUs held by it are charged while idle. *Open:* GPUs and hours per session.

## Milestones and verification gates

1. **Data:** load GPQA Diamond and print 5 formatted prompts per mode exactly as sent (`python3 -m src.data --show 5`). All gold letters check out. **Done 2026-11-13** (198 questions).
2. **Unit tests pass:** `python3 tests/run_all.py`. **Done** (130 pass with `.venv/bin/python`, 2026-11-17). **2026-11-27** (after setup job 3680134, the simulated smoke run and the cuDNN fix): 209 pass, 0 skip (with `transformers`; 207 before the job-3686748 fixes). **2026-11-24, second review (data check):** 199 pass, 0 skip (with `transformers` and the 4B `config.json` cached). **After the first review fixes:** 194 pass, 1 skip (with `transformers`); 183 pass, 12 skip with plain `python3`. Earlier the same day, with the training port: 173 pass, 1 skip (4B `config.json` not in the local HF cache) using a Python with `transformers`; 162 pass, 12 skip with plain `python3`. This repo has no `.venv/` yet (the run used Experiment-3's).
3. **Template parity:** `src/prompts.build_text` equals `tokenizer.apply_chat_template(...)` for every mode (run on DeltaAI or locally with the tokenizer files). **Done 2026-11-13**: 792/792 prompts (198 questions × 4 samples) match in both modes, as text and as token IDs (`transformers` 5.23.0, tokenizer @ `15852e8c`, chat_template sha256 `273d8e0e683b8850…`). **reinit, 2026-11-15:** 792/792 match (official thinking-off + `"<analysis>\n"`).
4. **Environment:** SGLang runs on a DeltaAI node and generates one answer per mode (see `INFRASTRUCTURE.md`).
5. **Pilot:** 20 questions × 3 modes × 1 sample (60 generations). The user reads several full outputs per mode: right prefix, sensible length, an `"answer": "X"` field at the end. Measure tokens per second and project the full run's GPU-hours.
6. **Full control run**, then a summary table (accuracy with confidence intervals, token counts, truncation rate, extraction failures) and a run note. **Done 2026-11-16:** `control_v1`, job 3611928, 0.94 SU, 2,376/2,376, 0 errors, check_run PASS. v3 accuracy think 0.563 · default 0.364 · reinit 0.313; lenient 0.564 · 0.374 · 0.485.

**Training gates (planned, each needs the user's approval):**

7. **Plan approved:** model, DAPO parts, async flavour and GPU split, code-server approach (vault `decisions/2026-11-22-training-setup-plan.md`). **Partly done 2026-11-22:** Qwen3.5-4B, synchronous disaggregated 2 + 2 GPUs, code-server via OnDemand check then own job. Still open: DAPO values; code-server GPUs/hours.
8. **Components ported and tested locally:** every training component above, with unit tests passing and no slime or GPU needed (`.venv/bin/python tests/run_all.py`). **Smoke-test part done 2026-11-24** (long-run and evaluation jobs remain).
9. **Code-server job:** starts on a compute node, reachable through the tunnel, `nvidia-smi` shows the requested GPUs; stops cleanly at the time limit.
10. **Smoke training (disaggregated):** 2 DAPO steps, step 2 generated with the weights after update 1 (proves the trainer -> SGLang weight push across GPUs), checkpoint saved and converted.
11. **Prompts + control final:** the new frozen `reinit` prompt in `src/prompts.py`, the matching control run done. Only then the real training run.

## Logging (one JSONL record per generation)

`run_id, mode, model, model_revision, qid, sample_idx, seed, option_order, chat_template_hash, input_text, input_token_ids, output_text, output_token_ids, output_logprobs, finish_reason, n_input_tokens, n_output_tokens, n_think_tokens, extracted_answer, extraction_rule, gold, correct, sampling_params, sglang_version, wall_time_s, error`

Extra generation fields include `n_analysis_tokens` and `analysis_closed` (reinit). The generator writes everything except `extracted_answer, extraction_rule, correct` to `runs/<run>/generations.jsonl` (list in `src/schema.py`). `src/score.py` adds those three, plus `extraction_detail, outcome, truncated, scoring_rule_version`, in a separate `scored.jsonl`; the raw file is never edited.

## Notes

Write a short note for every run (job id, cost, result), every decision (what, why, who decided, date) and every change. Never put personal paths, emails or GPQA question text in notes or commits.

**Where notes go:** the Obsidian vault **Research**, folder **`Reinitialized Thinking Experiment/`** (the vault's absolute path is in the gitignored `CLAUDE.local.md`, because it contains personal details). Never edit outside that folder or touch `.obsidian/`. Folders: `decisions/`, `changes/`, `components/`, `runs/`, plus `index.md` (link every new note from it). Frontmatter: `type, date, status, where (local|deltaai), job, git, related, tags`. Body: Summary · What · How it connects · Why · Results (real numbers only) · Risks & undo (`> [!warning]`) · Next steps. Notes from earlier work stay in `Past Experiments/GAR-Rebuild/` and are linked, not copied.

**References:** GAR official code `RickySkywalker/GAR-Official` @ `aa04bed`; training base `vaxope/Reinitialized-Thinking-Experiment-3` @ `97e4706`; slime `THUDM/slime` @ `79aa41a`; DAPO paper note in the vault at `Research Papers/Notes/DAPO.md`.
