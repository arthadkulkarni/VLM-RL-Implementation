# Run write-ups

Each run gets its own folder, named `<run-name>_<YYYY-MM-DD>/`, containing:

- `README.md`: setup, how it ran (SLURM jobs, failures, fixes), metrics,
  and what to look at next
- `metrics.csv`: per-step metrics parsed from the trainer logs
- any small artifacts worth keeping (generated data samples without images,
  filter summaries)

Checkpoints and full logs stay on DeltaAI. Each write-up says where they are.

| Date | Write-up | Summary |
|---|---|---|
| 2026-10-05 | [mini_run_2026-10-05](mini_run_2026-10-05/README.md) | First full questioner/solver loop. Needed 4 resumed jobs. The questioner collapses to reward −1 at steps 3–4 |
| 2026-10-05 | [mini_e2e_run_2026-10-05](mini_e2e_run_2026-10-05/README.md) | Same loop from scratch in one job (1h20m). The questioner collapse reproduces. Reward judging dominates questioner time |
