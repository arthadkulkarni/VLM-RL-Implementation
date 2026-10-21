# RISE mini run, end to end in one job (Qwen3-VL-8B)

**Status:** completed on 2026-10-05 in a single SLURM job (3317242, 1h20m,
exit 0). One big round (2 micro iterations × 2 steps) ran from the base model
through to a merged solver checkpoint at global step 4, with no resuming.

The [first mini run](../mini_run_2026-10-05/README.md) needed four jobs and
fixes between them. This run checks that the fixed pipeline runs cleanly from
scratch. It is still a **pipeline/integration test**: the batches are tiny and
there are only 4 steps per role, so the metrics are not evidence of learning.

## Setup

Same configuration as the first mini run, in a fresh storage directory.

| | |
|---|---|
| Base model | Qwen3-VL-8B-Instruct (questioner and solver both start from it) |
| Hardware | DeltaAI `ghx4`, 1 node (gh140), 4× GH200 120GB |
| Schedule | `BIG_ROUNDS=1`, `MICRO_ITERS=2`, `MICRO_STEPS=2` → global steps 2 and 4 per role |
| Batch | `rollout_batch_size=8`, `rollout.n=4` |
| Questioner data | 64 rows from 1 video graph (`0032_order_322_t89_5min.json`) |
| GPU split | questioner trains on GPUs 0–1 with 2 judge vLLM servers on 2–3; solver trains on all 4 |
| Script | `scripts_Qwen3-VL-8B/mini_run.sbatch` → `main.sh` |
| Storage | `/work/nvme/bffz/akulkarni8/rise/storage_RISE_mini_e2e` |
| Code | commit 2ab333b (eef87c4, the HEAD at submit time, only added docs) |

## How it ran

| Attempt | Job | Result |
|---|---|---|
| 1 | 3311069 | Failed after 2.7 min: the rank-0 liveness check queried the Ray dashboard (port 8265), which was not running. Fixed in 2ab333b |
| 2 | 3314483 | Failed after 46 min: `SafetensorError ... Disk quota exceeded (os error 122)` while saving a merged checkpoint on nvme |
| 3 | 3317242 | **Completed** in 4767 s |

Timeline of job 3317242:

| Stage | Wall clock (approx.) | Output |
|---|---|---|
| Questioner steps 1–2 | 19:31 → 20:17 | `Qwen3-VL-8B-Instruct_q_b1/global_step_2` |
| Generate + filter solver data | 20:17 → 20:23 | 30 questions |
| Solver steps 1–2 | 20:23 → 20:30 | `Qwen3-VL-8B-Instruct_s_b1/global_step_2` |
| Questioner steps 3–4 | 20:30 → 20:37 | `Qwen3-VL-8B-Instruct_q_b1/global_step_4` |
| Generate + filter solver data | 20:37 → 20:43 | 17 questions |
| Solver steps 3–4 | 20:43 → 20:50 | `Qwen3-VL-8B-Instruct_s_b1/global_step_4` |

## Metrics

Per-step values are in [`metrics.csv`](metrics.csv), parsed from the trainer's
metric dumps in the job log.

| Role | Step | Reward (overall) | Notes |
|---|---|---|---|
| Questioner | 1 | −0.202 | validity 0.75, format 0.75, skill_match 0.75, 48/48 candidates judged |
| Questioner | 2 | −0.202 | identical reward metrics to step 1; grad_norm 20 |
| Solver | 1 | 0.531 (accuracy) | resp. len 473 |
| Solver | 2 | 0.531 | resp. len 500 |
| Questioner | 3 | −1.000 | **validity 0, format 0, 0 candidates, grad_norm 0** |
| Questioner | 4 | −1.000 | same |
| Solver | 3 | 0.656 | resp. len 502 |
| Solver | 4 | 0.750 | resp. len 494 |

Speed: questioner steps 1–2 took 1389 s and 1097 s. Of that, 1269 s and 978 s
was spent waiting on the reward (the judge vLLM servers scoring 48 candidates).
Questioner steps 3–4 took about 95 s because there were no candidates to judge.
Solver steps took 55–67 s. Peak GPU memory allocated was 43 GB (questioner
steps 1–2), 64 GB (questioner steps 3–4) and 28 GB (solver).

## Solver training data produced by the questioner

| Built from | Passed validity check | Kept by majority vote | Skill | Answers (0/1) | Score range | Distinct queries |
|---|---|---|---|---|---|---|
| Questioner step 2 | 15/32 | 30/735 | 100% `bounded` | 21 / 9 | 0.56–0.78 | 11 |
| Questioner step 4 | 13/32 | 17/637 | 100% `bounded` | 9 / 8 | 0.56–0.78 | 6 |

Majority vote keeps candidates whose score falls in [0.3, 0.8]. Almost all
rejections were `out_of_band`. The kept questions are in
`solver_questions_step{2,4}.jsonl`, with images removed. The filter summaries
are in `solver_data_step{2,4}_summary.json`.

## Compared with the first mini run

| | First mini run (4 jobs) | This run (1 job) |
|---|---|---|
| Questioner reward, steps 1–2 | −0.93 | −0.20 |
| Questioner reward, steps 3–4 | −1.00 | −1.00 |
| Solver accuracy, steps 1→4 | 0.94 → 0.59 | 0.53 → 0.75 |
| Solver questions (step 2 / step 4) | 181 / 14 | 30 / 17 |
| Wall clock | spread over 09-24 to 10-05 | 1h20m |

## Things to look at before scaling up

1. **The questioner still collapses at steps 3–4.** This is the same failure
   as in the first run. Every rollout gets reward −1 (validity 0, format 0,
   no candidates). The group has no reward variance, so `pg_loss` and
   `grad_norm` are 0 and the questioner gets no learning signal. It happened
   in both runs after the first questioner update, so it looks systematic
   rather than bad luck.
2. **Questioner steps 1 and 2 have identical reward metrics.** Every reward
   field and the response length (mean 43) match exactly, even though the
   policy was updated between them (pg_loss −0.009, grad_norm 20). This may
   mean the step-2 rollouts repeat step 1's, for example because the sampling
   seed is fixed and there is only one video graph. It needs checking.
3. **There is no skill diversity in the solver data.** Every kept question is
   declared `bounded`, although the questioner's reward metrics show 75% of
   its step-1 rollouts as `sequential`. Several kept queries read as
   sequential ("What event occurs immediately after..."). The skill label may
   be lost or overwritten between generation and upload.
4. **The reward computation dominates questioner time.** It takes about 20
   min per step at this tiny batch size. It will need batching or more judge
   capacity at full scale.
5. **The nvme quota is tight.** Attempt 2 died on disk quota while saving a
   checkpoint. A full run needs shard cleanup (`cleanup_model_shards.sh`) or
   storage on hdd.
6. Solver accuracy went from 0.53 to 0.75. That is 32 rollouts per step on
   different questions each micro iteration, so it is noise-level.

## Where the artifacts are

These paths are on DeltaAI and are not in git.

- Merged HF checkpoints:
  `/work/nvme/bffz/akulkarni8/rise/storage_RISE_mini_e2e/models/Qwen3-VL-8B-Instruct_{q,s}_b1/global_step_{2,4}/actor/huggingface`
- Generated questions: `.../storage_RISE_mini_e2e/generated_question/`
- Full log: `logs/mini_run/rise-mini-e2e_3317242.log` in the repo checkout
  (gitignored)
- Offline wandb run: `wandb/offline-run-20261005_204603-wemn5sx6` (not synced)
