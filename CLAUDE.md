# CLAUDE.md: GAR reinitialized-thinking rebuild, control experiment

## Current goal (keep scope to this)

**Produce the control-group outputs:** an untrained Qwen model answering **GPQA Diamond** in three modes, **`think`**, **`default`** and **`reinit`**, generated with **SGLang on NCSA DeltaAI**, with every input, output token and error logged. These outputs are the baseline every trained model is compared against, so they must be exactly reproducible and fully transparent.

**Next run (2026-11-19):** the same control experiment on **Qwen3.5-4B** (the existing code and the completed run `control_v1` use Qwen3.5-2B), with prompts that add a few worked examples so the model writes the answer exactly as asked, and a reworded `reinit` prompt so that `</analysis>` closes before the answer. The prompts are designed together with the training side (training uses the same `reinit` prompt) and frozen before any run. New prompts or a new model mean new run names (e.g. `control_4b_v1`); `control_v1` stays as it is.

**Out of scope here:** training (it lives in the training repository). This repository holds only the control pipeline and the analysis of its outputs.

**Where things run:**
- **DeltaAI:** all model generation (SGLang), through `scripts/deltaai/submit.sh` only.
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
| `src/generate.py` | SGLang client: sends token ids to `/generate`, logs every generation, resumable, SIGTERM-safe. (Its `--checkpoint` option serves a trained model and needs `src/check_trained.py`, which lives in the training repository; it is not used here.) |
| `src/check_run.py` | mechanical checks of a run (prompts rebuilt, token ids, counts, completeness); PASS/FAIL |
| `src/score.py` | scoring + all metrics (v3 accuracy, lenient accuracy, strict v2); never edits the raw log |
| `src/compare_runs.py` | two runs side by side per mode: accuracy with clustered CI, lenient accuracy, truncation, token statistics |
| `src/compare_trajectories.py` | trajectory comparison per mode: answer-line shapes, commitments, invented sources, lengths, fixed/broken question IDs (counts and IDs only) |
| `viewer/index.html` | offline dashboard: tiles, filters, qid search, full trajectory, mode compare |
| `tests/` | `python3 tests/run_all.py` (no pytest needed); `tests/fake_run.py` makes a synthetic run with no GPQA text; `tests/fake_sglang.py` stands in for SGLang |
| `scripts/deltaai/` | `setup.sh` (login node, once), `submit.sh` (the only way to submit: smoke / pilot / full), `smoke_test.sbatch`, `run_control.sbatch`, `serve_and_generate.sh`, `common.sh`, `env.example.sh` (copy to gitignored `env.sh`) |

Everything uses the standard library only (Python 3.12), except the template-parity test, which needs `transformers` (and `jinja2`) in `.venv/`. Run the full suite with `.venv/bin/python tests/run_all.py`.

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
- any DeltaAI job (`sbatch`, `salloc`, `srun`). State the GPUs, the time requested and the estimated GPU-hours;
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

Qwen3.5's chat template **turns thinking off by default**: thinking is on only when `enable_thinking=True`. `src/prompts.py` builds every input by hand, matching the template (`chat_template.jinja` @ `15852e8c`) exactly; all prompt text lives there.

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

## Milestones and verification gates

1. **Data:** load GPQA Diamond and print 5 formatted prompts per mode exactly as sent (`python3 -m src.data --show 5`). All gold letters check out. **Done 2026-11-13** (198 questions).
2. **Unit tests pass:** `python3 tests/run_all.py`. **Done** (130 pass with `.venv/bin/python`, 2026-11-17).
3. **Template parity:** `src/prompts.build_text` equals `tokenizer.apply_chat_template(...)` for every mode (run on DeltaAI or locally with the tokenizer files). **Done 2026-11-13**: 792/792 prompts (198 questions × 4 samples) match in both modes, as text and as token IDs (`transformers` 5.23.0, tokenizer @ `15852e8c`, chat_template sha256 `273d8e0e683b8850…`). **reinit, 2026-11-15:** 792/792 match (official thinking-off + `"<analysis>\n"`).
4. **Environment:** SGLang runs on a DeltaAI node and generates one answer per mode (see `INFRASTRUCTURE.md`).
5. **Pilot:** 20 questions × 3 modes × 1 sample (60 generations). The user reads several full outputs per mode: right prefix, sensible length, an `"answer": "X"` field at the end. Measure tokens per second and project the full run's GPU-hours.
6. **Full control run**, then a summary table (accuracy with confidence intervals, token counts, truncation rate, extraction failures) and a run note. **Done 2026-11-16:** `control_v1`, job 3611928, 0.94 SU, 2,376/2,376, 0 errors, check_run PASS. v3 accuracy think 0.563 · default 0.364 · reinit 0.313; lenient 0.564 · 0.374 · 0.485.

## Logging (one JSONL record per generation)

`run_id, mode, model, model_revision, qid, sample_idx, seed, option_order, chat_template_hash, input_text, input_token_ids, output_text, output_token_ids, output_logprobs, finish_reason, n_input_tokens, n_output_tokens, n_think_tokens, extracted_answer, extraction_rule, gold, correct, sampling_params, sglang_version, wall_time_s, error`

Extra generation fields include `n_analysis_tokens` and `analysis_closed` (reinit). The generator writes everything except `extracted_answer, extraction_rule, correct` to `runs/<run>/generations.jsonl` (list in `src/schema.py`). `src/score.py` adds those three, plus `extraction_detail, outcome, truncated, scoring_rule_version`, in a separate `scored.jsonl`; the raw file is never edited.

## Notes

Write a short note for every run (job id, cost, result) and every decision (what, why, who decided, date), and share it with the training side. Never put personal paths, emails or GPQA question text in notes or commits.

**References:** GAR official code `RickySkywalker/GAR-Official` @ `aa04bed`.
