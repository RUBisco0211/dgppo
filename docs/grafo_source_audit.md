# GRAFO source audit (arXiv:2610.10302v1)

Update after the single-task discrepancy audit: the physics now advances
position with the old velocity, then applies damping and force; reaching
agents are snapped to their targets and frozen. Policy node coordinates are
relative to the visible landmark, and the actor concatenates local node
features with a four-round UniMP representation after two input feature
layers. These changes were checked by environment tests and a one-update PPO
smoke run. The table below preserves the initial discrepancies for provenance;
the action encoding, numerical reach threshold, bonus scale, and exact GRAFO
observation graph remain unverified because the authors' environment code is
unavailable. An optional five-action discrete policy/environment path was
added after the failed full-sequence runs; it follows the cited JaxInforMARL
Formation default, while the original continuous path remains available for
controlled comparisons. The Figure 9 graph edges and offsets were visually checked
against `media/drawio/GRAFO_tasks.drawio.pdf` in the arXiv source package.

Primary sources: [paper and appendix](https://arxiv.org/html/2610.10302v1), [cited JaxInforMARL repository](https://github.com/jselvaraaj/JaxInforMARL/tree/6659678355668a90e3afb41e3bf51b553a4ea2b9), and [original InforMARL Formation scenario](https://github.com/nsidn98/InforMARL/blob/7d3c42a66006991ec2c3462e065f0449317d5470/baselines/mpnn/mpe/mape/multiagent/scenarios/simple_formation.py). The paper authors' GRAFO implementation was not available in these sources. Accordingly, matching a cited repository's defaults does **not** establish that the authors' GRAFO experiments used those defaults.

## Confirmed matches

- The paper's Appendix, “Single-Task Description,” defines a task by adjacency `A`, target shifts `g`, an optional structurally equivalent node set, and optional static features. The implementation has those fields and only exposes one reference landmark; target shifts are used for reward and evaluation, not direct policy input. This matches the paper's qualitative description.
- The five sequence definitions in `dgppo/env/mpe/grafo.py` match Appendix “GRAFO Tasks” TS, TS-inc, TS-dec, CS, and GFS task names and order.
- Figure 9 target shifts for GRAFO-3 and GRAFO-4(1–4) match the current catalog. The other tasks' shifts appear consistent with extracted figure labels, though graph topology should be checked visually from the PDF rather than inferred from extracted text.
- The paper's hyperparameter table says episode length 50 and collisions disabled; the current environment uses 50 and returns zero collision cost. It also specifies 128-step rollout, 32 parallel environments, and 64 evaluation episodes.

## Concrete differences or unsupported assumptions

| Area | Current repository | Primary-source evidence | Consequence |
| --- | --- | --- | --- |
| Action space | Continuous two-axis `TanhNormal` policy and force | The cited JaxInforMARL `TargetMPEEnvironment` defaults to `DISCRETE_ACT`; `_discrete_action_to_control_input` maps five actions (stay/left/right/down/up) to axis forces. Its continuous handler is unimplemented (`pass`). Paper says GRAFO action space equals Formation, but does not state the action encoding. | Major fidelity gap; training difficulty and reachable trajectories differ. Do not claim proof that paper used discrete actions, but investigate this first. |
| Physics step | `velocity = 0.75*v + 5*a*0.1; position += velocity*0.1` | Cited JaxInforMARL `_integrate_state` first updates `position += old_velocity*dt`, then damps velocity and adds force. | One-step movement and 50-step control problem differ, even if force scale and damping match. |
| Reaching behavior | Agent can move after reaching; reward bonus uses first reach; evaluation tests whether all simultaneously enter radius. | Cited JaxInforMARL `_step` masks control of an agent already overlapping its landmark and snaps newly reached agents to goal. The paper itself only specifies a one-time bonus and team reaching targets, not snapping or terminal rules. | Potentially explains low final-step success; status is source ambiguity rather than a proven paper mismatch. |
| Reward scale | `-mean(Euclidean distances) + mean(newly reached)` with assumed bonus 1. | Appendix says *each agent's* reward is negative distance to assigned target plus one-time bonus. It does not specify coefficient, shared-reward aggregation, bonus size, clipping, or Euclidean vs squared. Cited JaxInforMARL uses global **sum of squared distances** with coefficient 1 and bonus 5 (in its training config); original InforMARL Formation uses a clipped mean Euclidean-distance penalty. | Current reward scale and objective remain unverified. Avoid choosing a reward merely because it is in a related environment. |
| Initialization | Agents uniform in `[-1,1]^2`, one landmark uniform in `[-0.5,0.5]^2`; no minimum separation. | Original InforMARL Formation uses those ranges. Cited JaxInforMARL training config uses radius 1 and rejects landmark pairs closer than 0.5, but has one landmark here. Paper says random initial positions only. | Ranges are plausible, but GRAFO distribution cannot be established from paper. |
| Graph observations | Base `MPE.get_graph` exposes absolute position and velocity of each agent and landmark as node features and relative state on task edges. All agents have reference landmark edges; no self edges among agents. | Paper says all agents access the reference position and graph topology; Appendix says observation/action spaces match Formation. Cited JaxInforMARL constructs **per-agent relative** node features and distance-based visibility edges, typically with self edges. The authors necessarily altered its graph structure for GRAFO, but no exact code is published. | Current topology/feature interface is a design interpretation, not exact sourced implementation. Absolute coordinates differ from original InforMARL Formation's local observation of own velocity, own position, and relative landmark. |
| Goal tolerance | `dist2goal = 0.05` | Original InforMARL Formation uses 0.05. Paper gives no numeric tolerance; cited JaxInforMARL uses overlap of radii 0.15 + 0.2 = 0.35. | A strict 0.05 simultaneous success criterion may dramatically depress measured success. Cannot equate it to paper metric without author code. |

## Recommended resolution order

1. Verify the catalog's edges visually against Figure 9 (PDF source is under `media/drawio/GRAFO_tasks.drawio.pdf` in the arXiv source package). Add source-coordinate fixtures/tests for graph and shifts.
2. Decide whether the training baseline should follow the cited JaxInforMARL discrete actions and dynamics, or the original InforMARL Formation MPE. Report this explicit branch choice: the paper alone does not settle it.
3. Run small controlled ablations of goal threshold, bonus scale, and reward aggregation with identical seeds. Label them as sensitivity analyses until the authors publish their environment code.
4. Avoid comparing the current 42.19% success to the paper's single-task result as if success definitions were identical. The paper's `R_{i,j}` is percentage of solved episodes, but exact radius/snap rules are absent.

## 2026-10-12 failed-run investigation

- Full seed-0 batch-512 FROG+EWC(M) runs failed to learn several new tasks.
  FT forgot earlier tasks, but this alone does not explain the failures.
- A 64-episode evaluation of the CS stage-1 GRAFO-6(1) checkpoint found
  that the last two agents reached their targets in only about 22% and 17%
  of episodes, respectively. Their median closest distances were about 0.83
  and 0.87. The zero team success is therefore not merely a strict 0.05
  threshold artifact.
- A target-aware proportional controller solved all 64 sampled episodes on
  GRAFO-5, GRAFO-6(1), GRAFO-7, GRAFO-4(2), and GRAFO-4(3) in 50 steps.
  The implemented dynamics can reach those formations.
- The continuous-action GRAFO-6(1) single-task run reached 0/64 success
  after 1,003,520 frames. Removing the extra actor `ScaleHid` layer and
  limiting each task head to the final distribution layer also yielded 0/64
  after the same budget. Those fixes align the action module with the paper's
  two-hidden-layer specification but are not yet an empirical learning fix.
- The cited JaxInforMARL Formation code defaults to five discrete actions
  (stay, left, right, down, up); its continuous-action converter is empty.
  The paper does not say whether its GRAFO fork retained that default.
  `--action-encoding discrete` implements this as a controlled fidelity
  check. The same cited Formation code uses overlap of 0.15 and 0.20 entity
  radii; `--reach-threshold 0.35` is the corresponding controlled setting,
  not a verified GRAFO parameter.
- With five discrete actions, a target-aware controller solved 0/64 sampled
  GRAFO-6(1) episodes at radius 0.05 but 64/64 at radii 0.15, 0.25, and
  0.35. The discrete-action radius-0.35 five-million-frame single-task
  validation is in progress in tmux session `d6`; its 200-update evaluation
  reached 6/64 team successes, whereas both continuous-action 245-update
  diagnostic runs had 0/64. This comparison changes the action distribution
  and reach threshold together, so it does not isolate either choice.
- Re-evaluating the discrete-action 300-update checkpoint with the same 64
  episode keys at thresholds 0.05, 0.15, 0.25, and 0.35 yielded 0%, 1.56%,
  4.69%, and 64.06% team success. Early performance is highly sensitive to
  this undocumented GRAFO threshold. Do not equate the radius-0.35 result
  with the paper's solved percentage until the authors' criterion is known.
- The five-million-frame GRAFO-6(1) single-task run completed 1221 PPO
  updates and reached 60/64 (93.75%) successes at radius 0.35. At the final
  checkpoint, a fixed 64-episode sensitivity evaluation yielded 0%, 4.69%,
  48.44%, and 89.06% success at radii 0.05, 0.15, 0.25, and 0.35.
  Consequently, the source-informed configuration is learnable, but the
  paper's reported 99.4% single-task median cannot yet be compared directly.
- A separate GRAFO-4(2) run with five discrete actions and radius 0.35
  reached 64/64 successes after 245 PPO updates (1,003,520 frames). This
  establishes that the previously failed GFS task is individually learnable
  under the source-informed setting; it does not establish that FROG will
  learn it after another task. A continuous-action radius-0.35 GRAFO-6(1)
  ablation is in progress to separate the action and threshold effects.
  Fixed-key evaluation of the final GRAFO-4(2) policy at radii 0.05, 0.15,
  0.25, and 0.35 gave 0%, 15.63%, 85.94%, and 100% success, respectively.
- The controlled continuous-action radius-0.35 GRAFO-6(1) run reached 61/64
  (95.31%) team successes after 245 updates (1,003,520 frames); it had
  already reached 64/64 at update 200. With the same two-hidden-layer actor,
  continuous actions at radius 0.05 had 0/64 at update 245. Thus, the
  threshold is a major driver of the original single-task failure. This is
  still an inference about our implementation, not proof of the authors'
  unknown GRAFO radius.
- A two-task GFS prefix, GRAFO-4(1) then GRAFO-4(2), is being trained with
  continuous actions, radius 0.35, FROG+EWC(M), and one million frames per
  task. It finished with 100% success on GRAFO-4(1) and 0% on GRAFO-4(2),
  so the larger radius alone does not fix new-task learning after switching.
  The reported two-task AFP/AP are both 0.5, with AF 0 only because the
  second task was never learned. A continuous-action, radius-0.35
  GRAFO-4(2) single-task run reached 62/64 (96.88%) after the same one
  million frames, confirming that the failure is specific to the sequential
  setup. The matching FT run finished with 13/64 (20.31%) on the old task
  and 62/64 (96.88%) on the new one: it learned the new task while forgetting
  much of the old one. This points to FROG freezing and/or EWC as the source
  of plasticity loss. A FROG-without-EWC run with the isolated minibatch RNG
  finished at 64/64 on the first task and 62/64 (96.88%) on the second task,
  with AP=AFP=98.44% and AF=0. This shows the frozen encoder and separate
  output heads can learn this sequence without EWC. The matched FROG+EWC(M)
  run with isolated RNG retained 64/64 on the old task but learned 0/64 on
  the new task. Its final EWC penalty was about 0.0094, while the final PPO
  loss was about -0.0017. These scalar losses do not alone prove a gradient
  imbalance, but the paired outcome identifies EWC as the cause of failed
  new-task learning in this implementation and budget. A coefficient-100
  continuation from the exact stage-1 checkpoint also finished at 0/64
  new-task success while preserving 64/64 old-task success. This is a
  hyperparameter sensitivity result; it does not establish an error in the
  diagonal-Fisher formula. With coefficient 1, the new task reached 64/64,
  but the old task declined to 57/64 (89.06%). Thus coefficient tuning can
  restore plasticity in this two-task run, but it did not outperform FROG
  alone (old 64/64, new 62/64). The paper's coefficient is retained as the
  default and these lower values remain explicitly labeled ablations.
- The 0.05-threshold, continuous-action GRAFO-6(1) single-task checkpoint
  after 1,003,520 frames was reevaluated on 64 episodes. It solved 0/64;
  the median *best-over-time team maximum* target distance was 0.784 and
  even its closest team state was never within 0.35. Thus the failure is
  more than missing a strict final 0.05 precision window. The first four
  agents sometimes approached their goals, whereas the last two had mean
  individual closest distances about 0.234 and 0.241. This is a one-seed,
  short-budget diagnosis, not a five-million-frame verdict. The continuation
  reached 5,001,216 total frames and 1221 PPO updates at radius 0.05. Its
  final training-loop evaluation solved 4/64 (6.25%) episodes. An independent
  fixed-key evaluation of the final checkpoint solved 2/64 (3.125%); 42/64
  trajectories came within 0.35 at some point, but only 14/64 within 0.15.
  The median closest-over-time team maximum distance was 0.258. This remains
  far from the paper's reported single-task result under our stricter,
  unverified success criterion. Because the original checkpoint predated
  the isolated minibatch RNG change, this continuation used a new dedicated
  shuffle stream and is not an exact uninterrupted same-seed run. A manual
  resume-state migration initially retained the completed-stage evaluation
  matrix and caused an IndexError at the final update; the matrix was reset
  in the cloned update-1200 checkpoint and the final 21 updates completed.
- Prior CGMARL runs used the process-global NumPy RNG to shuffle PPO
  minibatches. Despite equal first rollouts, W&B initialization could cause
  first-update minibatch order to differ between runs. New runs keep a
  dedicated NumPy shuffle RNG in the checkpoint and no longer use the global
  RNG for PPO minibatches. Earlier FT/FROG+EWC runs are qualitative controls;
  exact paired comparisons must be rerun with the isolated RNG version.
- The user-requested batch size 512 differs from the paper's 32 minibatches
  per 4096-step update (batch size 128). With four PPO epochs, this changes
  the actor/critic optimizer steps from 128 to 32 per update. The EWC
  coefficient listed in the paper was tuned under that different update
  schedule, so the coefficient ablations above cannot be read as evidence
  that the paper's hyperparameter is intrinsically wrong.

## 2026-10-13 paired threshold runs

Two full GFS-sequence runs were launched with FROG+EWC(M), explicit EWC
coefficient 1, seed 0, continuous actions, batch size 512, 32 environments,
and 5 million requested frames per task. They differ only in the reach
threshold: 0.05 (`g05`, GPU 2) versus 0.35 (`g35`, GPU 1). Each writes
`metrics.jsonl`, `evaluation.csv`, checkpoints, GIFs, and W&B data under its
own directory in `logs/cgmarl/full_cl/`. AP is defined at the first task end;
AF and AFP require at least two completed tasks. Threshold changes both the
one-time bonus/freeze dynamics and the success criterion, so this is a
training-environment comparison, not merely re-scoring a common policy.
