# GAR reinitialized-thinking rebuild: control run and training setup

Two parts:

1. **Control (done for Qwen3.5-2B, next on Qwen3.5-4B):** an untrained Qwen model answering **GPQA Diamond** (198 multiple-choice questions) in three modes, with every input, output token and error logged. This is the baseline every trained model is compared against.
2. **Training (being set up, not run yet):** train the model in the `reinit` format with **DAPO** on SuperGPQA physics/chemistry/biology questions, using slime with rollout and training on **separate GPUs** (synchronous, 2 rollout + 2 training, no colocation), then evaluate it exactly like the control.

| Mode | Prompt | Assistant turn starts with | Model writes |
|---|---|---|---|
| `think` | short task line + question | `<think>\n` | hidden reasoning, `</think>`, then the answer |
| `default` | same as `think` | `<think>\n\n</think>\n\n` | the answer directly |
| `reinit` | system persona + step-by-step analysis instructions + question | `<think>\n\n</think>\n\n<analysis>\n` | analysis, `</analysis>`, then the answer |

`reinit` ("reinitialized thinking") turns Qwen's built-in thinking off and has the model reason in a plain-text `<analysis>` block instead. All modes get the same question, the same shuffled options, the same sampling settings and the same 81,920-token output cap. Every prompt ends with Qwen's recommended instruction: ``Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.``

## Control results (`control_v1`, Qwen3.5-2B, 198 questions × 4 samples per mode)

| | Think | Default | Reinit |
|---|---|---|---|
| Accuracy (rule v3) | 56.3% | 36.4% | 31.3% |
| Accuracy, lenient | 56.4% | 37.4% | 48.5% |

Job 3611928, 0.94 SU, 2,376/2,376 generations, 0 errors, `check_run` PASS.

## Files

| File | Purpose |
|---|---|
| `configs/control.toml` | every control-run parameter, with its source |
| `src/data.py` | pinned GPQA CSV to `data/gpqa_diamond.jsonl` |
| `src/prompts.py` | exact model input per mode (all prompt text lives here); option shuffle |
| `src/extract.py` | answer extraction: rule `last_answer_field_v3` (strict field or answer line) and the lenient reading |
| `src/generate.py` | sends prompts to SGLang, logs every generation (runs on DeltaAI) |
| `src/check_run.py` | pass/fail checks of a finished run |
| `src/score.py` | scoring and metrics; `src/schema.py` lists the log fields |
| `src/compare_runs.py`, `src/compare_trajectories.py` | two runs side by side: accuracy, tokens, answer shapes, fixed/broken question IDs |
| `scripts/deltaai/` | setup, job submission and Slurm job scripts |
| `viewer/index.html` | offline dashboard for browsing every generation |
| `tests/` | `run_all.py` runs the tests; `fake_sglang.py` stands in for SGLang; `fake_run.py` makes a synthetic run |

Training files (data build, slime flags, reward, slime image, DeltaAI training and code-server jobs) are planned and listed in `CLAUDE.md` under *Training components (planned)*. They are ported from `vaxope/Reinitialized-Thinking-Experiment-3` one at a time.

## Usage

```bash
.venv/bin/python tests/run_all.py                 # tests (template parity needs transformers in .venv)
python3 -m src.data                               # build data/gpqa_diamond.jsonl
python3 -m src.data --show 5                      # print prompts exactly as sent
scripts/deltaai/submit.sh smoke|pilot|full        # on DeltaAI (after setup.sh)
python3 -m src.check_run runs/<run>               # pass/fail checks
python3 -m src.score runs/<run>/generations.jsonl # -> scored.jsonl, summary.md
open viewer/index.html                            # load scored.jsonl
```

## Status

Control:
- [x] Data prepared and gold letters verified
- [x] Unit tests pass
- [x] Prompts match the official chat template (text and token IDs)
- [x] DeltaAI setup, smoke test, pilot (`pilot_v1`, job 3611388)
- [x] Full run and summary (`control_v1`, job 3611928)
- [ ] Qwen3.5-4B control with the new prompts (worked examples, reworded `reinit`)

Training:
- [ ] Plan approved: model size, DAPO parts, GPU layout (synchronous, 2 + 2), code-server
- [ ] Training components ported from Experiment-3 and tested locally
- [ ] Code-server job on DeltaAI
- [ ] Smoke training run (2 DAPO steps, separate rollout and training GPUs)
- [ ] Training run, then evaluation against the matching control

**GPQA is test-only:** never train on it or commit its text. `data/` and `runs/` are gitignored.
