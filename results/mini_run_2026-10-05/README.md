# RISE mini run: first full questioner/solver loop (Qwen3-VL-8B)

**Status:** completed on 2026-10-05. Every stage of one big round (2 micro
iterations × 2 steps) trained, saved a checkpoint, and merged to HuggingFace
format.

This was a **pipeline/integration test**, not a training experiment. The batches
are tiny (8 prompts × 4 rollouts) and there are only 4 steps per role, so the
metrics below show the loop runs end to end. They are not evidence of learning.

## Setup

| | |
|---|---|
| Base model | Qwen3-VL-8B-Instruct (questioner and solver both start from it) |
| Hardware | DeltaAI `ghx4`, 1 node, 4× GH200 120GB |
| Schedule | `BIG_ROUNDS=1`, `MICRO_ITERS=2`, `MICRO_STEPS=2` → global steps 2 and 4 per role |
| Batch | `RISE_ROLLOUT_BATCH_SIZE=8`, `RISE_GLOBAL_BATCH_SIZE=4`, `RISE_ROLLOUT_N=4`, `QUESTION_NUM_SAMPLES_PER_MICRO=8` |
| Questioner data | 64 rows from 1 video graph (`questioner_videos_mini.parquet`) |
| GPU split | questioner trains on 2 GPUs (FSDP), solver on 4 |
| Script | `scripts_Qwen3-VL-8B/mini_run.sbatch` → `main.sh` |

## How it ran

The run resumes from existing checkpoints, so the four stages finished in
separate SLURM jobs. Each job failed later on, for reasons fixed in the commits
listed below. The next job then picked up from the last finished stage.

| Stage | Job | Date | Output |
|---|---|---|---|
| Questioner steps 1–2 | 3207357 | 09-24 | `Qwen3-VL-8B-Instruct_q_b1/global_step_2` |
| Solver steps 1–2 | 3207658 | 09-25 | `Qwen3-VL-8B-Instruct_s_b1/global_step_2` |
| Questioner steps 3–4 | 3218885 | 09-27 | `Qwen3-VL-8B-Instruct_q_b1/global_step_4` |
| Solver steps 3–4 | 3314482 | 10-05 | `Qwen3-VL-8B-Instruct_s_b1/global_step_4` |

Fixes committed between the jobs:
- f1f6c7c Trainer fixes for 4-GPU nodes; freeze vision tower for text-only questioner
- 3c817b9 Fix early training stop and merge path in train scripts
- b741b35 Fix ray.init startup timeouts in mini run
- f28aeaf Load the nvme venv in all sbatch scripts via use_env.sh
- 267b1ce Prefer venv NCCL over HPC SDK NCCL; fail fast when a vLLM server dies
- 2ab333b Check rank-0 worker death without the Ray dashboard

The final job ran at commit 2ab333b. Earlier stages ran on earlier commits,
starting from 8ed97a4 and the commits right after it.

## Metrics

Per-step values are in [`metrics.csv`](metrics.csv), parsed from the trainer logs.

| Role | Step | Reward (overall) | Notes |
|---|---|---|---|
| Questioner | 1 | −0.928 | validity 0.5, skill_match 0.5, 32/32 candidates judged |
| Questioner | 2 | −0.926 | same |
| Solver | 1 | 0.938 (accuracy) | resp. len 487 |
| Solver | 2 | 0.750 | resp. len 441 |
| Questioner | 3 | −1.000 | **validity 0, format 0, 0 candidates** |
| Questioner | 4 | −1.000 | same |
| Solver | 3 | 0.719 | resp. len 555 |
| Solver | 4 | 0.594 | resp. len 553 |

Speed: about 450 s per questioner step at steps 1–2 and 110–135 s at steps 3–4.
Solver steps took 45–75 s. Peak GPU memory allocated was 43–65 GB for the
questioner and 28 GB for the solver.

## Solver training data produced by the questioner

| Built from | Questions kept | Skill distribution | Answers (0/1) | Score range |
|---|---|---|---|---|
| Questioner step 2 | 181 | 100% `bounded` | 118 / 63 | 0.56–0.78 |
| Questioner step 4 | 14 | 100% `bounded` | 10 / 4 | 0.56–0.78 |

The full kept questions are in `solver_questions_step{2,4}.jsonl`, with images
removed. The filter summaries are in `solver_data_step{2,4}_summary.json`.

## Things to look at before scaling up

1. **The questioner collapsed at steps 3–4.** Every rollout got reward −1
   (validity 0, format 0, no candidates). Identical rewards within a GRPO
   group give zero advantage, so `pg_loss` is 0 and the questioner gets no
   learning signal. Steps 1–2 were already close to −1.
2. **There is no skill diversity.** All generated questions use the `bounded`
   skill. None of the other 7 categories (causal, dynamic, identity, negative,
   sequential, static, synchronous) appear.
3. **The solver's data shrank.** It went from 181 to 14 questions between the
   two micro iterations, because of item 1.
4. Solver accuracy fell from 0.94 to 0.59. With batches this small that is
   mostly noise, and the questions differ from step to step.

## Where the artifacts are

These paths are on DeltaAI and are not in git.

- Merged HF checkpoints:
  `/work/hdd/bffz/akulkarni8/rise/storage_RISE_mini/models/Qwen3-VL-8B-Instruct_{q,s}_b1/global_step_{2,4}/actor/huggingface`
  (copied from `/work/nvme/.../storage_RISE_mini`)
- Full logs: `logs/mini_run/rise-mini_{3207357,3207658,3218885,3314482}.log`
  in the repo checkout on DeltaAI. They are gitignored.
