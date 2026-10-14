# GRAFO continual learning audit

Source: [GRAFO paper and appendix](https://arxiv.org/html/2610.10302v1),
especially the empirical setting, task identities, metrics, and hyperparameter
table. The authors' GRAFO/CL source has not been located, so EWC and MAS
normalization cannot be verified exactly.

## Implemented sequence

`train_grafo_continual.py` iterates through TS, CS, GFS, TS-inc, or TS-dec.
Each task gets 5 million environment frames by default. At a task boundary,
the shared actor and critic transfer to the next task; the actor receives a
new output head. Evaluation restores each previous task's head and graph,
forming the task-by-task success matrix for AFP, AF, and AP. Multi-task
observations carry one-hot task identity on agent nodes.

## Methods

| Method | Current mechanism |
| --- | --- |
| FT | Shared actor and critic continue without regularization. |
| L2 | Penalizes distance of shared actor weights from the preceding task's final weights. |
| EWC (M) | Uses the squared per-sample gradient of sampled action log probability as empirical diagonal Fisher; stores one anchor/importance matrix per prior task and sums their penalties. |
| EWC (O) | Uses the latest anchor and a running importance matrix, `0.9 × old + new`. |
| MAS (M) | Uses the absolute per-sample gradient of mean squared policy mode; stores one anchor/importance matrix per prior task. |
| MAS (O) | Uses the latest anchor and the same decayed aggregation as EWC (O). |
| FROG | Freezes the graph encoder after the first task, alongside any method above. |

The critic and task-specific output head are not regularized. Importance is
estimated from 64 independent trajectories of up to 50 valid steps each.
Steps after early success are masked from the estimate. Paper coefficients
are selected by method, online/multi mode, and FROG flag unless overridden.

## Verification and limits

- A two-task CPU run completed for EWC, L2, and MAS+FROG, including task
  switches, prior-task evaluation, checkpoints, and final metrics. These
  one-update runs verify execution only, not learning or retention.
- Targeted checks confirm one-hot task input, positive finite EWC/MAS
  importance, positive L2/EWC/MAS penalties after parameter movement, and
  FROG's encoder gradient mask.
- Initial full-sequence experiments exposed new-task learning failures. A
  later six-run, seed-0 comparison completed on TS, CS, and GFS with a 0.35
  reach threshold and an explicit FROG+EWC(M) coefficient of 1. Exact
  normalization of the authors' EWC/MAS scores and the scope of their
  task-specific heads remain unverified without their implementation.
- Runs with `--batch-size 512` use 8 PPO minibatches per epoch instead of
  the paper's 32 and must be reported as a throughput variant.

## Full-sequence run records

`scripts/run_grafo_cl_full.sh` launches a full TS, CS, or GFS sequence with
5 million frames per task and batch size 512. The six completed comparison
runs used FROG+EWC(M) or FT on each setting. Each run writes `metrics.jsonl`,
checkpoints, evaluation GIFs, and a W&B `eval/video` upload at each evaluation.
The trainer evaluates every 100 PPO updates and at the end of each task,
including all previously learned tasks. The trainer calls
`scripts/export_grafo_cl_csv.py` after each evaluation to update
`evaluation.csv`; `stage_complete=1` marks task-end rows, which
include AP and, from the second task onward, AF and AFP.

| Setting | Method | AP | AF | AFP |
| --- | --- | ---: | ---: | ---: |
| TS | FROG+EWC(M), coefficient 1 | 0.8802 | -0.0531 | 0.8359 |
| TS | FT | 0.9870 | -0.8625 | 0.2682 |
| CS | FROG+EWC(M), coefficient 1 | 0.9438 | -0.1484 | 0.8250 |
| CS | FT | 0.9688 | -0.9297 | 0.2250 |
| GFS | FROG+EWC(M), coefficient 1 | 0.9844 | -0.0104 | 0.9766 |
| GFS | FT | 0.9922 | -0.9844 | 0.2539 |

All six runs used continuous actions, threshold 0.35, batch size 512, and
one seed. FT has no regularization even though its run name includes
`lambda1`. These are controlled comparisons within this implementation,
not the paper's 10-seed numerical reproduction. Their per-run CSV files are
under `logs/cgmarl/full_cl/` and W&B project `cgmarl`.

## Follow-up organization

- [ ] Move the CGMARL-specific environment, algorithm, training entry point,
  scripts, tests, and documentation into one clearly scoped package; update
  imports and commands without changing the training behavior.
- [ ] Run the relevant checks against the reorganized package and confirm
  checkpoints and result-reading paths still work.
- [ ] Keep `logs/` in its current location; do not move training outputs into
  the package or include them in the commit.
