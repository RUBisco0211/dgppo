"""External rollout collection for the stateful GPUDrive simulator."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ..env.gpudrive.adapter import GPUDriveDGPPOAdapter
from ..utils.graph import GraphsTuple
from ..utils.typing import Array


class GPUDriveRollout(NamedTuple):
    graph: GraphsTuple
    critic_graph: GraphsTuple
    actions: Array
    rewards: Array
    goal_progress_reward: Array
    goal_reward: Array
    vehicle_collision_reward: Array
    offroad_reward: Array
    lane_deviation_reward: Array
    goal_distance: Array
    lane_deviation: Array
    costs: Array
    next_graph: GraphsTuple
    next_critic_graph: GraphsTuple
    next_costs: Array
    min_clearance: Array
    terminated: Array
    truncated: Array
    alive_mask: Array
    log_pis: Array
    goal_achieved: Array
    native_collision: Array
    road_collision: Array
    controlled_collision: Array
    background_collision: Array
    episode_end: Array
    episode_length: Array


def _stack_graphs(graphs):
    return jtu.tree_map(
        lambda *values: None if values[0] is None else jnp.stack(values),
        *graphs,
        is_leaf=lambda value: value is None,
    )


class GPUDriveRolloutRunner:
    """Collect fixed-length chunks without placing GPUDrive in ``jax.jit``."""

    def __init__(self, adapter: GPUDriveDGPPOAdapter, collect_costs: bool = True,
                 graph_fn=None):
        self.adapter = adapter
        self.collect_costs = collect_costs
        self.shared_graph = graph_fn is not None
        self.graph_fn = graph_fn or adapter.local_graph_batch
        self.alive = np.ones(
            (adapter.num_worlds, adapter.config.num_controlled), dtype=bool
        )
        self.scene_steps = np.zeros((adapter.num_worlds,), dtype=np.int32)

    def reset_masks(self):
        if self.adapter._controlled_valid is None:
            self.alive.fill(True)
        else:
            self.alive[:] = self.adapter._controlled_valid.detach().cpu().numpy()
        self.scene_steps.fill(0)

    def collect(
        self,
        policy_fn: Callable[[GraphsTuple, Array], tuple[Array, Array]],
        key: Array,
        horizon: int | None = None,
        frame_callback=None,
        progress_callback=None,
        progress_interval: int = 8,
    ) -> tuple[GPUDriveRollout, Array]:
        """Collect a chunk; ``policy_fn`` receives a batched local graph."""
        horizon = horizon or self.adapter.config.rollout_horizon
        records = []
        for step in range(horizon):
            key, action_key = jax.random.split(key)
            current_graph = self.graph_fn().graph
            critic_graph = current_graph if self.shared_graph else self.adapter.scene_graph_batch()
            if self.collect_costs:
                current_cost, _ = self.adapter.continuous_cost_batch()
            else:
                current_cost = jnp.zeros(
                    (self.adapter.num_worlds, self.adapter.config.num_controlled, 2),
                    dtype=jnp.float32,
                )
            goal_distance, lane_deviation = self.adapter.task_state_batch()
            actions, log_pis = policy_fn(current_graph, action_key)
            actions = jnp.where(jnp.asarray(self.alive)[..., None], actions, 0.0)

            rewards_all, dones_all, infos_all = self.adapter.step(actions)
            if frame_callback is not None:
                frame_callback()
            next_graph = self.graph_fn().graph
            next_critic_graph = next_graph if self.shared_graph else self.adapter.scene_graph_batch()
            if self.collect_costs:
                next_cost, next_clearance = self.adapter.continuous_cost_batch()
            else:
                next_cost = jnp.zeros_like(current_cost)
                next_clearance = jnp.zeros(current_cost.shape[:-1], dtype=jnp.float32)
            next_goal_distance, next_lane_deviation = self.adapter.task_state_batch()

            slots = self.adapter._controlled_slots().cpu().numpy()
            world_rows = np.arange(self.adapter.num_worlds)[:, None]
            rewards_np = rewards_all[world_rows, slots].detach().cpu().numpy()
            dones_np = dones_all[world_rows, slots].detach().cpu().numpy().astype(bool)
            info_np = infos_all.detach().cpu().numpy()
            controlled_info = info_np[world_rows, slots]
            road_collision = controlled_info[..., 0] > 0
            vehicle_collision = controlled_info[..., 1:3].sum(axis=-1) > 0
            collision_done = road_collision | vehicle_collision
            goal_done = controlled_info[..., 3] > 0
            # GPUDrive supplies the configured IPPO reward unchanged.
            shaped_rewards = jnp.nan_to_num(jnp.asarray(rewards_np))
            goal_reward = float(self.adapter.env.config.goal_achieved_weight) * jnp.asarray(
                goal_done, dtype=jnp.float32)
            goal_progress_reward = jnp.zeros_like(shaped_rewards)
            vehicle_collision_reward = float(self.adapter.env.config.collision_weight) * jnp.asarray(
                controlled_info[..., 1:3].sum(axis=-1), dtype=jnp.float32
            )
            offroad_reward = float(self.adapter.env.config.off_road_weight) * jnp.asarray(
                road_collision, dtype=jnp.float32
            )
            lane_deviation_reward = jnp.zeros_like(shaped_rewards)
            cc, cb = self.adapter.collision_categories(infos_all)
            # Match the original DGPPO MPE/LiDAR tasks: collisions are
            # continuous safety costs but do not terminate the policy
            # trajectory. Goal completion remains a true terminal event.
            terminated = dones_np & goal_done
            alive_before = self.alive.copy()
            self.alive &= ~terminated
            self.scene_steps += 1
            truncated_world = self.scene_steps >= int(self.adapter.env.episode_len)
            truncated = np.broadcast_to(truncated_world[:, None], terminated.shape) & self.alive
            world_was_alive = alive_before.any(axis=1)
            episode_end = (world_was_alive & (~self.alive).all(axis=1)) | truncated_world
            episode_length = np.where(episode_end, self.scene_steps, 0)

            records.append(GPUDriveRollout(
                current_graph, critic_graph, actions, shaped_rewards,
                goal_progress_reward,
                goal_reward, vehicle_collision_reward, offroad_reward,
                lane_deviation_reward,
                next_goal_distance, next_lane_deviation, current_cost,
                next_graph, next_critic_graph, next_cost, next_clearance,
                jnp.asarray(terminated), jnp.asarray(truncated),
                jnp.asarray(alive_before), log_pis, jnp.asarray(goal_done),
                jnp.asarray(collision_done), jnp.asarray(road_collision),
                jnp.asarray(cc.cpu().numpy()),
                jnp.asarray(cb.cpu().numpy()),
                jnp.asarray(episode_end), jnp.asarray(episode_length),
            ))
            # Do not inject a freshly reset episode into the same fixed-length
            # rollout. Keeping terminal worlds absorbing makes the DGPPO
            # return target stop exactly at the terminal transition. The
            # training/eval entry point resets every world before the next
            # collect call.
            if progress_callback is not None and (
                (step + 1) % progress_interval == 0 or step + 1 == horizon
            ):
                progress_callback(step + 1, horizon)
        rollout = jtu.tree_map(
            lambda *values: None if values[0] is None else jnp.stack(values, axis=1),
            *records,
            is_leaf=lambda value: value is None,
        )
        # A GNN-IPPO rollout contains the flattened road context for every
        # vehicle and time step. Keeping the complete sequence in JAX's GPU
        # allocator makes the next rollout coexist with retained PPO scratch
        # buffers and eventually exhausts VRAM. Store the replay batch on the
        # host; update() transfers only one world-minibatch at a time.
        return jax.device_get(rollout), key
