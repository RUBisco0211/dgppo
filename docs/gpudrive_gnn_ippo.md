# GPUDrive JAX GNN-IPPO

`train_gpudrive_gnn_ippo.py` implements IPPO with DGPPO's JAX graph encoder
while retaining GPUDrive's checked-in SB3-IPPO task definition.

## Environment and reward

- Every valid vehicle with an unfinished route is controlled; 64 is only the
  compiled padding capacity.
- Non-vehicles are removed, matching the project decision for this comparison.
- Reward is the SB3-IPPO weighted combination: goal `+1`, vehicle collision
  `-0.5`, and off-road `-0.5`.
- Collision behavior is `ignore`; reaching the goal terminates only that car.
- The default entry samples the four checked-in example scenarios. A dataset
  directory and scene prefix can be supplied explicitly for larger datasets.

## Graph observation

- Vehicle slots and identities remain fixed within an episode.
- Every ordered pair within 50 m has an edge; there is no top-k vehicle
  selection and no distance smoothing heuristic.
- Edges outside the radius and padding slots are routed to the padding node.
- Edge features use receiver-local relative position, relative speed and
  relative heading.
- Each vehicle embeds its own official-style road observation (up to 200 road
  records, 13 public features plus one internal validity bit). Road records are
  not graph nodes.
- Actor and critic use independent Graph Transformer parameters. The critic
  returns one local value per vehicle, and GAE/advantages remain per-agent.

## Action modes

Discrete, matching the official 13 by 7 action grid:

```bash
python train_gpudrive_gnn_ippo.py --action-mode discrete
```

Continuous, using a tanh-Gaussian acceleration/steering head:

```bash
python train_gpudrive_gnn_ippo.py --action-mode continuous
```

Both modes use the same graph, reward, PPO target and logging path. Generated
run names include the action mode, for example
`gnn_ippo_discrete_seed000_<timestamp>_<id>`.
