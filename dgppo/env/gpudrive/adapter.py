"""Runtime adapter around GPUDrive's stateful Torch/CUDA simulator."""

from dataclasses import dataclass
import itertools
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ...utils.typing import Array
from .config import GPUDriveDGPPOConfig
from .geometry import continuous_collision_cost, road_edge_collision_cost
from .graph import (
    EgoGraphBatch, build_actor_scene_graph, build_ego_graph, build_ippo_scene_graph,
    build_scene_graph,
)


_build_ego_graph_jit = jax.jit(build_ego_graph, static_argnames=("config",))
_build_scene_graph_jit = jax.jit(build_scene_graph, static_argnames=("config",))
_build_ippo_scene_graph_jit = jax.jit(
    build_ippo_scene_graph, static_argnames=("config",)
)
_continuous_collision_cost_jit = jax.jit(
    continuous_collision_cost, static_argnames=("margin_m", "mode")
)
_road_edge_collision_cost_jit = jax.jit(
    road_edge_collision_cost, static_argnames=("margin_m", "mode")
)


def _build_ego_graph_batch(ego, partner, partner_valid, partner_controlled,
                           road, config):
    return jax.vmap(
        lambda a, b, c, d, e: build_ego_graph(a, b, c, d, e, config=config)
    )(ego, partner, partner_valid, partner_controlled, road)


def _build_scene_graph_batch(vehicles, valid, controlled_mask, road, speeds, config):
    return jax.vmap(
        lambda a, b, c, d, e: build_scene_graph(a, b, c, d, e, config=config)
    )(vehicles, valid, controlled_mask, road, speeds)


def _build_actor_scene_graph_batch(vehicles, valid, controlled_mask, road, speeds, config):
    return jax.vmap(
        lambda a, b, c, d, e: build_actor_scene_graph(
            a, b, c, d, e, config=config
        )
    )(vehicles, valid, controlled_mask, road, speeds)


def _build_ippo_scene_graph_batch(vehicles, valid, road, speeds, config):
    return jax.vmap(
        lambda a, b, c, d: build_ippo_scene_graph(a, b, c, d, config=config)
    )(vehicles, valid, road, speeds)


def _continuous_collision_cost_batch(ego_position, ego_heading, ego_length, ego_width,
                                     other_position, other_heading, other_length,
                                     other_width, valid_mask, margin_m, mode):
    return jax.vmap(
        lambda ep, eh, el, ew, op, oh, ol, ow, vm: continuous_collision_cost(
            ep, eh, el, ew, op, oh, ol, ow, vm, margin_m, mode)
    )(ego_position, ego_heading, ego_length, ego_width, other_position,
      other_heading, other_length, other_width, valid_mask)


_build_ego_graph_batch_jit = jax.jit(
    _build_ego_graph_batch, static_argnames=("config",)
)
_build_scene_graph_batch_jit = jax.jit(
    _build_scene_graph_batch, static_argnames=("config",)
)
_build_actor_scene_graph_batch_jit = jax.jit(
    _build_actor_scene_graph_batch, static_argnames=("config",)
)
_build_ippo_scene_graph_batch_jit = jax.jit(
    _build_ippo_scene_graph_batch, static_argnames=("config",)
)
_continuous_collision_cost_batch_jit = jax.jit(
    _continuous_collision_cost_batch, static_argnames=("margin_m", "mode")
)


def _road_edge_collision_cost_batch(samples, length, width, margin_m, mode):
    return jax.vmap(
        lambda s, l, w: road_edge_collision_cost(s, l, w, margin_m, mode)
    )(samples, length, width)


_road_edge_collision_cost_batch_jit = jax.jit(
    _road_edge_collision_cost_batch, static_argnames=("margin_m", "mode")
)


@dataclass(frozen=True)
class PreparedWorld:
    kept_ids: tuple[int, ...]
    controlled_ids: tuple[int, ...]
    background_ids: tuple[int, ...]
    layout_center: tuple[float, float]


def _to_jax(tensor: torch.Tensor):
    """Share a contiguous Torch tensor with JAX through DLPack."""
    return jax.dlpack.from_dlpack(tensor.contiguous())


def _finite_torch(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _lane_deviation_torch(roadmap: torch.Tensor) -> torch.Tensor:
    """Distance from the ego origin to the nearest RoadLane line segment."""
    center = roadmap[..., :2]
    length = roadmap[..., 2].clamp_min(0.0)
    heading = roadmap[..., 5]
    direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    along = torch.clamp(
        -(center * direction).sum(dim=-1), min=-length / 2, max=length / 2
    )
    closest = center + along[..., None] * direction
    distance = torch.linalg.norm(closest, dim=-1)
    lane = (roadmap[..., 6].round().to(torch.int64) == 3) & (length > 0)
    return torch.where(lane, distance, torch.inf).amin(dim=-1)


def _road_clearance_torch(samples: torch.Tensor, length: torch.Tensor,
                          width: torch.Tensor, mode: str) -> torch.Tensor:
    """Torch equivalent of the RoadEdge clearance used for initial filtering."""
    depth = samples[..., 0]
    valid = (depth > 0) & (samples[..., 1].round().to(torch.int64) == 1)
    direction = samples[..., 2:4] / depth.clamp_min(1e-6)[..., None]
    half_length = 0.5 * length[..., None]
    half_width = 0.5 * width[..., None]
    if mode == "oriented_box":
        support = direction[..., 0].abs() * half_width + direction[..., 1].abs() * half_length
    elif mode == "ellipse":
        support = torch.sqrt(
            torch.square(direction[..., 0] * half_width)
            + torch.square(direction[..., 1] * half_length)
        )
    else:
        raise ValueError(f"unknown geometry mode: {mode}")
    clearance = torch.where(valid, depth - support, torch.inf)
    return clearance.amin(dim=-1)


class GPUDriveDGPPOAdapter:
    """Keep exactly five controlled and ten free scene vehicles.

    The wrapped object is a ``GPUDriveTorchEnv``. Environment stepping remains
    outside JAX jit; observations are transferred to JAX without a CPU copy
    when Torch and JAX use the same device.
    """

    def __init__(self, env, config: GPUDriveDGPPOConfig = GPUDriveDGPPOConfig()):
        self.env = env
        self.config = config
        self.prepared_worlds: List[PreparedWorld] = []
        self._kept_slots = None
        self._scene_valid = None
        self._controlled_slot_cache = None
        self._controlled_valid = None
        self._vehicle_pool_initialized = False

    @property
    def num_worlds(self) -> int:
        return self.env.num_worlds

    def prepare_scenes(self, seed: int = 0, fixed_eval: bool = False) -> List[PreparedWorld]:
        """Reset and choose a compact straight-road five-car task per world.

        The first call establishes a 15-vehicle interaction pool. Training
        uses ``seed`` to vary the five controlled roles inside that pool;
        evaluation uses the deterministic best layout.
        """
        if self.config.control_all_vehicles:
            return self._prepare_all_controlled()
        self.env.sim.reset(list(range(self.num_worlds)))
        local = self.env.sim.self_observation_tensor().to_torch().clone()
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch().clone()
        roadmap = self.env.sim.agent_roadmap_tensor().to_torch().clone()
        lidar = self.env.sim.lidar_tensor().to_torch().clone()
        prepared = []
        delete_by_world = {}

        for world in range(self.num_worlds):
            ids = local[world, :, 7]
            lane_deviation = _lane_deviation_torch(roadmap[world])
            scale = float(getattr(self.env.config, "agent_size_scale", 1.0))
            road_clearance = _road_clearance_torch(
                lidar[world, :, 1], absolute[world, :, 10] * scale,
                absolute[world, :, 11] * scale, self.config.geometry,
            )
            physical = (ids >= 0) & (local[world, :, 1] > 0)
            valid = (
                physical
                & torch.isfinite(lane_deviation)
                & (lane_deviation <= self.config.initial_lane_max_distance_m)
                & (road_clearance >= self.config.safety_margin_m)
            )
            valid_indices = torch.where(valid)[0]
            if valid_indices.numel() < self.config.num_controlled:
                raise ValueError(
                    f"world {world} has {valid_indices.numel()} valid vehicles; "
                    f"the baseline requires at least {self.config.num_controlled}"
                )

            goal_distance = torch.linalg.norm(
                absolute[world, valid_indices, :2]
                - absolute[world, valid_indices, 8:10],
                dim=-1,
            )
            task_indices = valid_indices[
                goal_distance > float(self.env.config.dist_to_goal_threshold)
            ]
            if task_indices.numel() < self.config.num_controlled:
                raise ValueError(
                    f"world {world} has only {task_indices.numel()} vehicles with an "
                    f"unfinished navigation task; {self.config.num_controlled} are required"
                )

            controlled_indices = self._select_straight_group(
                task_indices, absolute[world], seed, world, fixed_eval
            )
            if controlled_indices is None:
                raise ValueError(
                    f"world {world} has no valid compact straight-road five-car layout"
                )
            physical_indices = torch.where(physical)[0]
            other = physical_indices[~torch.isin(physical_indices, controlled_indices)]
            if not self._vehicle_pool_initialized and not self.config.keep_all_scene_vehicles:
                # Keep ten free cars whose current positions are closest to
                # one of the five controlled routes.  This gives interaction
                # traffic instead of ten arbitrary cars elsewhere on the map.
                point = absolute[world, other, :2]
                start = absolute[world, controlled_indices, :2]
                finish = absolute[world, controlled_indices, 8:10]
                segment = finish - start
                rel = point[:, None, :] - start[None, :, :]
                projection = (rel * segment[None]).sum(-1) / torch.square(
                    torch.linalg.norm(segment, dim=-1).clamp_min(1e-6)
                )[None]
                projection = projection.clamp(0.0, 1.0)
                closest = start[None] + projection[..., None] * segment[None]
                route_distance = torch.linalg.norm(point[:, None] - closest, dim=-1).amin(dim=1)
                other = other[torch.argsort(route_distance)[:self.config.num_background]]
            if (not self.config.keep_all_scene_vehicles
                    and other.numel() != self.config.num_background):
                raise ValueError(
                    f"world {world} requires exactly {self.config.num_background} free vehicles; "
                    f"found {other.numel()} after selecting controlled vehicles"
                )
            ordered = torch.cat([controlled_indices, other])
            kept_ids = tuple(int(v) for v in ids[ordered].tolist())
            controlled_ids = tuple(int(v) for v in ids[controlled_indices].tolist())
            background_ids = tuple(int(v) for v in ids[other].tolist())
            layout_center = absolute[world, controlled_indices, :2].mean(dim=0)
            prepared.append(PreparedWorld(
                kept_ids, controlled_ids, background_ids,
                tuple(float(value) for value in layout_center.tolist()),
            ))
            if not self._vehicle_pool_initialized and not self.config.keep_all_scene_vehicles:
                kept_set = set(kept_ids)
                delete_by_world[world] = [
                    int(value) for value in ids[physical].tolist()
                    if int(value) not in kept_set
                ]

        if not self._vehicle_pool_initialized:
            if any(delete_by_world.values()):
                self.env.sim.deleteAgents(delete_by_world)
            self._vehicle_pool_initialized = True
            self.env.sim.reset(list(range(self.num_worlds)))

        local_after = self.env.sim.self_observation_tensor().to_torch()
        controlled_state = self.env.sim.controlled_state_tensor().to_torch()
        response_type = self.env.sim.response_type_tensor().to_torch()
        controlled_state.zero_()
        for world, selection in enumerate(prepared):
            ids = local_after[world, :, 7].round().to(torch.int64)
            present = local_after[world, :, 1] > 0
            for agent_id in selection.controlled_ids:
                matches = torch.where(present & (ids == agent_id))[0]
                if matches.numel() != 1:
                    raise RuntimeError(f"controlled vehicle ID {agent_id} is not unique in world {world}")
                controlled_state[world, matches[0], 0] = 1
                response_type[world, matches[0]] = 0  # Dynamic
            if self.config.background_behavior == "static":
                for agent_id in selection.background_ids:
                    matches = torch.where(present & (ids == agent_id))[0]
                    if matches.numel() != 1:
                        raise RuntimeError(
                            f"background vehicle ID {agent_id} is not unique in world {world}"
                        )
                    response_type[world, matches[0]] = 2  # Static

        self.env.cont_agent_mask = self.env.get_controlled_agents_mask()
        self.env.num_valid_controlled_agents_across_worlds = int(self.env.cont_agent_mask.sum().item())
        self.env.num_agents = self.env.num_valid_controlled_agents_across_worlds
        self.prepared_worlds = prepared
        scene_slots, scene_valid = [], []
        for world, selection in enumerate(prepared):
            ids = local_after[world, :, 7].round().to(torch.int64)
            present = local_after[world, :, 1] > 0
            rows = []
            for agent_id in selection.kept_ids:
                match = torch.where(present & (ids == agent_id))[0]
                if match.numel() != 1:
                    raise RuntimeError(f"vehicle ID {agent_id} is not unique in world {world}")
                rows.append(match[0])
            rows = torch.stack(rows)
            padded = torch.zeros(self.config.max_scene_vehicles, dtype=torch.long,
                                 device=rows.device)
            if rows.numel() > self.config.max_scene_vehicles:
                raise ValueError(
                    f"world {world} contains {rows.numel()} vehicles, exceeding "
                    f"max_scene_vehicles={self.config.max_scene_vehicles}"
                )
            padded[:rows.numel()] = rows
            scene_slots.append(padded)
            scene_valid.append(
                torch.arange(self.config.max_scene_vehicles, device=rows.device)
                < rows.numel()
            )
        self._kept_slots = torch.stack(scene_slots)
        self._scene_valid = torch.stack(scene_valid)
        self._controlled_slot_cache = self._kept_slots[:, :self.config.num_controlled]
        self._controlled_valid = torch.ones_like(
            self._controlled_slot_cache, dtype=torch.bool
        )
        return prepared

    def _prepare_all_controlled(self) -> List[PreparedWorld]:
        """Use GPUDrive's native non-trivial agents with fixed-shape padding."""
        self.env.sim.reset(list(range(self.num_worlds)))
        local = self.env.sim.self_observation_tensor().to_torch().clone()
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch().clone()
        if self.config.use_native_controlled_mask:
            native_mask = self.env.get_controlled_agents_mask().clone()
            prepared, scene_slots, scene_valid, controlled_slots, controlled_valid = [], [], [], [], []
            for world in range(self.num_worlds):
                ids = local[world, :, 7].round().to(torch.int64)
                physical = (ids >= 0) & (local[world, :, 1] > 0)
                controlled_rows = torch.where(native_mask[world] & physical)[0]
                background_rows = torch.where(physical & ~native_mask[world])[0]
                ordered = torch.cat([controlled_rows, background_rows])
                n_controlled, n_scene = controlled_rows.numel(), ordered.numel()
                padded_scene = torch.zeros(self.config.max_scene_vehicles, dtype=torch.long, device=local.device)
                padded_scene[:n_scene] = ordered
                padded_controlled = torch.zeros_like(padded_scene)
                padded_controlled[:n_controlled] = controlled_rows
                scene_slots.append(padded_scene)
                scene_valid.append(torch.arange(self.config.max_scene_vehicles, device=local.device) < n_scene)
                controlled_slots.append(padded_controlled)
                controlled_valid.append(torch.arange(self.config.num_controlled, device=local.device) < n_controlled)
                controlled_ids = tuple(int(value) for value in ids[controlled_rows].tolist())
                background_ids = tuple(int(value) for value in ids[background_rows].tolist())
                prepared.append(PreparedWorld(
                    controlled_ids + background_ids, controlled_ids, background_ids,
                    tuple(float(value) for value in absolute[world, controlled_rows, :2].mean(0).tolist()),
                ))
            self.env.cont_agent_mask = native_mask
            self.env.num_valid_controlled_agents_across_worlds = int(native_mask.sum().item())
            self.env.num_agents = self.env.num_valid_controlled_agents_across_worlds
            self.prepared_worlds = prepared
            self._kept_slots = torch.stack(scene_slots)
            self._scene_valid = torch.stack(scene_valid)
            self._controlled_slot_cache = torch.stack(controlled_slots)
            self._controlled_valid = torch.stack(controlled_valid)
            self._vehicle_pool_initialized = True
            return prepared
        controlled_state = self.env.sim.controlled_state_tensor().to_torch()
        controlled_state.zero_()
        prepared, scene_slots, scene_valid, controlled_slots, controlled_valid = [], [], [], [], []
        for world in range(self.num_worlds):
            ids = local[world, :, 7].round().to(torch.int64)
            physical = (ids >= 0) & (local[world, :, 1] > 0)
            physical_rows = torch.where(physical)[0]
            goal_distance = torch.linalg.norm(
                absolute[world, physical_rows, :2]
                - absolute[world, physical_rows, 8:10], dim=-1
            )
            task_rows = physical_rows[
                goal_distance > float(self.env.config.dist_to_goal_threshold)
            ]
            if task_rows.numel() == 0:
                raise ValueError(f"world {world} has no non-trivial controlled vehicles")
            if physical_rows.numel() > self.config.max_scene_vehicles:
                raise ValueError(
                    f"world {world} contains {physical_rows.numel()} vehicles, exceeding "
                    f"max_scene_vehicles={self.config.max_scene_vehicles}"
                )
            controlled_state[world, task_rows, 0] = 1
            background_rows = physical_rows[~torch.isin(physical_rows, task_rows)]
            ordered = torch.cat([task_rows, background_rows])
            n_controlled, n_scene = task_rows.numel(), ordered.numel()
            padded_scene = torch.zeros(
                self.config.max_scene_vehicles, dtype=torch.long, device=local.device
            )
            padded_scene[:n_scene] = ordered
            valid_scene = torch.arange(
                self.config.max_scene_vehicles, device=local.device
            ) < n_scene
            padded_controlled = torch.zeros_like(padded_scene)
            padded_controlled[:n_controlled] = task_rows
            valid_controlled = torch.arange(
                self.config.num_controlled, device=local.device
            ) < n_controlled
            scene_slots.append(padded_scene)
            scene_valid.append(valid_scene)
            controlled_slots.append(padded_controlled)
            controlled_valid.append(valid_controlled)
            controlled_ids = tuple(int(value) for value in ids[task_rows].tolist())
            background_ids = tuple(int(value) for value in ids[background_rows].tolist())
            prepared.append(PreparedWorld(
                controlled_ids + background_ids, controlled_ids, background_ids,
                tuple(float(value) for value in absolute[world, task_rows, :2].mean(0).tolist()),
            ))
        self.env.cont_agent_mask = self.env.get_controlled_agents_mask()
        self.env.num_valid_controlled_agents_across_worlds = int(
            self.env.cont_agent_mask.sum().item()
        )
        self.env.num_agents = self.env.num_valid_controlled_agents_across_worlds
        self.prepared_worlds = prepared
        self._kept_slots = torch.stack(scene_slots)
        self._scene_valid = torch.stack(scene_valid)
        self._controlled_slot_cache = torch.stack(controlled_slots)
        self._controlled_valid = torch.stack(controlled_valid)
        self._vehicle_pool_initialized = True
        return prepared

    @staticmethod
    def _angle_separation(a, b):
        return torch.abs(torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi)

    def _select_straight_group(self, task_indices, absolute, seed, world, fixed_eval):
        """Greedily choose five compact, spaced, same-direction valid routes."""
        positions = absolute[task_indices, :2]
        goals = absolute[task_indices, 8:10]
        routes = goals - positions
        lengths = torch.linalg.norm(routes, dim=-1)
        directions = routes / lengths[:, None].clamp_min(1e-6)
        cos_limit = float(np.cos(self.config.straight_max_heading_difference_rad))
        eligible = (lengths >= self.config.straight_min_goal_distance_m) & (
            lengths <= self.config.straight_max_goal_distance_m)
        anchors = torch.where(eligible)[0]
        if anchors.numel() == 0:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 7919 * world)
        order = anchors if fixed_eval else anchors[torch.randperm(anchors.numel(), generator=generator).to(anchors.device)]
        best = None
        best_score = float("inf")
        for anchor in order.tolist():
            same = eligible & ((directions @ directions[anchor]) >= cos_limit)
            delta = positions - positions[anchor]
            forward = directions[anchor]
            lateral_axis = torch.stack((-forward[1], forward[0]))
            longitudinal = torch.abs(delta @ forward)
            lateral = torch.abs(delta @ lateral_axis)
            distance = torch.linalg.norm(delta, dim=-1)
            compact = (
                (longitudinal <= self.config.straight_max_cluster_length_m)
                & (lateral <= self.config.straight_max_cluster_width_m)
            )
            candidates = torch.where(same & compact)[0]
            candidates = candidates[torch.argsort(distance[candidates])]
            selected = []
            for candidate in candidates.tolist():
                if selected:
                    clearance = torch.linalg.norm(positions[candidate] - positions[selected], dim=-1).min()
                    if float(clearance) < self.config.straight_min_vehicle_spacing_m:
                        continue
                selected.append(candidate)
                if len(selected) == self.config.num_controlled:
                    break
            if len(selected) == self.config.num_controlled:
                score = float(torch.linalg.norm(positions[selected] - positions[selected].mean(0), dim=-1).max())
                if not fixed_eval:
                    return task_indices[torch.tensor(selected, device=task_indices.device)]
                if score < best_score:
                    best_score, best = score, selected
        if best is None:
            return None
        return task_indices[torch.tensor(best, device=task_indices.device)]

    def _estimate_intersection_center(self, positions, goals):
        """Estimate the shared junction from intersections of recorded trip chords."""
        direction = goals - positions
        candidates = []
        for i, j in itertools.combinations(range(positions.shape[0]), 2):
            a, b = direction[i], direction[j]
            cross = a[0] * b[1] - a[1] * b[0]
            if float(torch.abs(cross)) < 0.15 * float(torch.linalg.norm(a) * torch.linalg.norm(b)):
                continue
            delta = positions[j] - positions[i]
            t = (delta[0] * b[1] - delta[1] * b[0]) / cross
            point = positions[i] + t * a
            if torch.isfinite(point).all():
                candidates.append(point)
        if not candidates:
            # GPUDrive recenters each source scenario around its map origin.
            # A fully parallel traffic snapshot contains no line intersection
            # from which to refine that origin.
            return torch.zeros(2, dtype=positions.dtype, device=positions.device)

        candidate_points = torch.stack(candidates)
        start_radius = torch.linalg.norm(positions[None] - candidate_points[:, None], dim=-1)
        goal_radius = torch.linalg.norm(goals[None] - candidate_points[:, None], dim=-1)
        start_vec = positions[None] - candidate_points[:, None]
        goal_vec = goals[None] - candidate_points[:, None]
        crossing = (start_vec * goal_vec).sum(dim=-1) < 0
        usable = (
            crossing
            & (start_radius >= self.config.intersection_min_start_radius_m)
            & (start_radius <= self.config.intersection_max_start_radius_m)
            & (goal_radius >= self.config.intersection_min_goal_radius_m)
            & (goal_radius <= self.config.intersection_max_goal_radius_m)
        )
        # Prefer the junction supported by the most routes, then the one with
        # the smallest robust start/goal radius.  This avoids using a vehicle
        # cluster centroid as a fictitious junction.
        support = usable.sum(dim=1)
        compactness = torch.where(
            usable, start_radius + goal_radius, torch.inf
        ).amin(dim=1)
        score = support.to(torch.float32) * 1000.0 - compactness
        return candidate_points[int(torch.argmax(score))]

    def _select_intersection_triplet(self, task_indices, positions, goals, center, seed, world):
        """Select routes with distinct incoming and outgoing junction arms."""
        if positions.shape[0] < self.config.num_controlled:
            return None
        valid_combos = []
        min_arm_angle = self.config.intersection_min_arm_angle_rad
        min_route_angle = self.config.intersection_min_route_angle_rad
        for combo in itertools.combinations(range(positions.shape[0]), self.config.num_controlled):
            p = positions[list(combo)]
            g = goals[list(combo)]
            start_vec = p - center
            goal_vec = g - center
            start_radius = torch.linalg.norm(start_vec, dim=-1)
            goal_radius = torch.linalg.norm(goal_vec, dim=-1)
            trip_length = torch.linalg.norm(g - p, dim=-1)
            if bool(torch.any(start_radius < self.config.intersection_min_start_radius_m)):
                continue
            if bool(torch.any(start_radius > self.config.intersection_max_start_radius_m)):
                continue
            if bool(torch.any(goal_radius < self.config.intersection_min_goal_radius_m)):
                continue
            if bool(torch.any(goal_radius > self.config.intersection_max_goal_radius_m)):
                continue
            if bool(torch.any(trip_length < self.config.intersection_min_trip_m)):
                continue
            if bool(torch.any(trip_length > self.config.intersection_max_trip_m)):
                continue
            start_angles = torch.atan2(start_vec[:, 1], start_vec[:, 0])
            goal_angles = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
            start_separations = []
            goal_separations = []
            for i, j in itertools.combinations(range(self.config.num_controlled), 2):
                start_separations.append(self._angle_separation(start_angles[i], start_angles[j]))
                goal_separations.append(self._angle_separation(goal_angles[i], goal_angles[j]))
            min_start_separation = torch.stack(start_separations).min()
            min_goal_separation = torch.stack(goal_separations).min()
            route_turn = self._angle_separation(start_angles, goal_angles)
            if float(min_start_separation) < min_arm_angle:
                continue
            if float(min_goal_separation) < min_arm_angle:
                continue
            if bool(torch.any(route_turn < min_route_angle)):
                continue
            score = float(min_start_separation + min_goal_separation) - 0.005 * float(trip_length.std())
            valid_combos.append((score, combo))
        if not valid_combos:
            return None
        valid_combos.sort(key=lambda item: item[0], reverse=True)
        if world == 0:
            # World zero is the stable, visually comparable evaluation layout.
            selected = valid_combos[0][1]
        else:
            # Other worlds diversify training while retaining the same map and
            # lane specifications. Randomness is deterministic for a run seed.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 7919 * world)
            pool_size = min(len(valid_combos), 16)
            selected = valid_combos[int(torch.randint(pool_size, (1,), generator=generator))][1]
        return task_indices[list(selected)]

    def task_state_batch(self):
        """Return goal distance and nearest lane-center distance for controlled cars."""
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        roadmap = self.env.sim.agent_roadmap_tensor().to_torch()
        worlds = torch.arange(self.num_worlds, device=absolute.device)[:, None]
        slots = self._controlled_slots()
        controlled = absolute[worlds, slots]
        goal_distance = torch.linalg.norm(controlled[..., :2] - controlled[..., 8:10], dim=-1)
        lane_deviation = _lane_deviation_torch(roadmap[worlds, slots])
        lane_deviation = torch.nan_to_num(
            lane_deviation, nan=self.config.observation_radius_m,
            posinf=self.config.observation_radius_m,
            neginf=self.config.observation_radius_m,
        )
        goal_distance = torch.nan_to_num(
            goal_distance, nan=self.config.observation_radius_m,
            posinf=self.config.observation_radius_m,
            neginf=self.config.observation_radius_m,
        )
        if self._controlled_valid is not None:
            goal_distance = torch.where(self._controlled_valid, goal_distance, 0.0)
            lane_deviation = torch.where(self._controlled_valid, lane_deviation, 0.0)
        return _to_jax(goal_distance), _to_jax(lane_deviation)

    def _controlled_slots(self) -> torch.Tensor:
        if self._controlled_slot_cache is not None:
            return self._controlled_slot_cache
        counts = self.env.cont_agent_mask.sum(dim=1)
        if not torch.all(counts == self.config.num_controlled):
            raise RuntimeError(f"each world must have exactly {self.config.num_controlled} controlled vehicles")
        return torch.stack([torch.where(mask)[0] for mask in self.env.cont_agent_mask])

    def local_graph_batch(self) -> EgoGraphBatch:
        """Build all world-local actor/Vh graphs in one batched GPU operation."""
        if self._kept_slots is None:
            raise RuntimeError("prepare_scenes must be called before requesting graphs")
        if self.config.control_all_vehicles:
            absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
            local = self.env.sim.self_observation_tensor().to_torch()
            roadmap = self.env.sim.agent_roadmap_tensor().to_torch()
            worlds = torch.arange(self.num_worlds, device=absolute.device)[:, None]
            vehicles = absolute[worlds, self._kept_slots].clone()
            speeds = local[worlds, self._kept_slots, 0].clone()
            vehicles = torch.where(self._scene_valid[..., None], vehicles, 0.0)
            speeds = torch.where(self._scene_valid, speeds, 0.0)
            vehicles[..., 10:13] *= float(
                getattr(self.env.config, "agent_size_scale", 1.0)
            )
            controlled_mask = self._controlled_valid
            road = roadmap[worlds, self._kept_slots]
            road_valid = (road[..., 6] > 0) & torch.isfinite(road[..., :7]).all(dim=-1)
            road_distance = torch.linalg.norm(road[..., :2], dim=-1)
            road_valid &= road_distance <= self.config.observation_radius_m
            order = torch.argsort(
                torch.where(road_valid, road_distance, torch.inf), dim=-1
            )[..., :self.config.num_road_segments]
            road = torch.gather(
                road, 2, order[..., None].expand(-1, -1, -1, road.shape[-1])
            )
            road = torch.where(
                torch.gather(road_valid, 2, order)[..., None], road, 0.0
            )
            return _build_actor_scene_graph_batch_jit(
                _to_jax(_finite_torch(vehicles)), _to_jax(self._scene_valid),
                _to_jax(controlled_mask), _to_jax(_finite_torch(road)),
                _to_jax(_finite_torch(speeds)), config=self.config,
            )
        ego_all = self.env.sim.self_observation_tensor().to_torch()
        partner_all = self.env.sim.partner_observations_tensor().to_torch()
        roadmap_all = self.env.sim.agent_roadmap_tensor().to_torch()
        worlds = torch.arange(self.num_worlds, device=ego_all.device)[:, None]
        slots = self._controlled_slots()
        ego = ego_all[worlds, slots].clone()
        scale = float(getattr(self.env.config, "agent_size_scale", 1.0))
        ego[:, :, 1:4] *= scale

        candidates = partner_all[worlds, slots].clone()
        candidate_ids = candidates[..., 8].round().to(torch.int64)
        self_ids = ego[..., 7].round().to(torch.int64)
        candidate_valid = (candidate_ids >= 0) & (candidate_ids != self_ids[..., None])
        candidate_distance = torch.linalg.norm(candidates[..., 1:3], dim=-1)
        candidate_valid &= candidate_distance <= self.config.observation_radius_m
        ranked_distance = torch.where(candidate_valid, candidate_distance, torch.inf)
        candidate_rows = torch.argsort(ranked_distance, dim=-1)[..., :self.config.max_neighbors]
        gather_index = candidate_rows[..., None].expand(-1, -1, -1, candidates.shape[-1])
        selected_partner = torch.gather(candidates, 2, gather_index)
        partner_valid = torch.gather(candidate_valid, 2, candidate_rows)
        selected_partner[:, :, :, 4:7] *= scale
        selected_partner = torch.where(partner_valid[..., None], selected_partner, 0.0)
        controlled_ids = ego[..., 7].round().to(torch.int64)
        selected_ids = selected_partner[..., 8].round().to(torch.int64)
        partner_controlled = (
            selected_ids[..., None] == controlled_ids[:, None, None, :]
        ).any(dim=-1) & partner_valid
        road = roadmap_all[worlds, slots]
        road_valid = (road[..., 6] > 0) & torch.isfinite(road[..., :7]).all(dim=-1)
        road_distance = torch.linalg.norm(road[..., :2], dim=-1)
        road_valid &= road_distance <= self.config.observation_radius_m
        road_order = torch.argsort(torch.where(road_valid, road_distance, torch.inf), dim=-1)
        road_order = road_order[..., :self.config.num_road_segments]
        road = torch.gather(road, 2, road_order[..., None].expand(-1, -1, -1, road.shape[-1]))
        selected_road_valid = torch.gather(road_valid, 2, road_order)
        road = torch.where(selected_road_valid[..., None], road, 0.0)
        return _build_ego_graph_batch_jit(
            _to_jax(_finite_torch(ego)), _to_jax(_finite_torch(selected_partner)), _to_jax(partner_valid),
            _to_jax(partner_controlled), _to_jax(_finite_torch(road)), config=self.config,
        )

    def ippo_graph_batch(self) -> EgoGraphBatch:
        """Return fixed-identity, all-in-radius vehicle graphs for GNN-IPPO."""
        if not self.config.control_all_vehicles:
            raise RuntimeError("GNN-IPPO requires control_all_vehicles=True")
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        local = self.env.sim.self_observation_tensor().to_torch()
        roadmap = self.env.sim.agent_roadmap_tensor().to_torch()
        worlds = torch.arange(self.num_worlds, device=absolute.device)[:, None]
        vehicles = absolute[worlds, self._kept_slots].clone()
        speeds = local[worlds, self._kept_slots, 0].clone()
        vehicles = torch.where(self._scene_valid[..., None], vehicles, 0.0)
        speeds = torch.where(self._scene_valid, speeds, 0.0)
        vehicles[..., 10:13] *= float(
            getattr(self.env.config, "agent_size_scale", 1.0))
        road = roadmap[worlds, self._kept_slots]
        road_valid = ((road[..., 6] > 0)
                      & torch.isfinite(road[..., :7]).all(dim=-1))
        road_distance = torch.linalg.norm(road[..., :2], dim=-1)
        road_valid &= road_distance <= self.config.observation_radius_m
        order = torch.argsort(
            torch.where(road_valid, road_distance, torch.inf), dim=-1
        )[..., :self.config.num_road_segments]
        road = torch.gather(
            road, 2, order[..., None].expand(-1, -1, -1, road.shape[-1]))
        road = torch.where(
            torch.gather(road_valid, 2, order)[..., None], road, 0.0)
        return _build_ippo_scene_graph_batch_jit(
            _to_jax(_finite_torch(vehicles)), _to_jax(self._scene_valid),
            _to_jax(_finite_torch(road)), _to_jax(_finite_torch(speeds)),
            config=self.config,
        )

    def scene_graph_batch(self):
        """Build all centralized task-critic graphs in one batched GPU operation."""
        if self._kept_slots is None:
            raise RuntimeError("prepare_scenes must be called before requesting graphs")
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        local = self.env.sim.self_observation_tensor().to_torch()
        roadmap = self.env.sim.agent_roadmap_tensor().to_torch()
        worlds = torch.arange(self.num_worlds, device=absolute.device)[:, None]
        vehicles = absolute[worlds, self._kept_slots].clone()
        speeds = local[worlds, self._kept_slots, 0].clone()
        vehicles = torch.where(self._scene_valid[..., None], vehicles, 0.0)
        speeds = torch.where(self._scene_valid, speeds, 0.0)
        vehicles[:, :, 10:13] *= float(getattr(self.env.config, "agent_size_scale", 1.0))
        road = roadmap[worlds, self._controlled_slots()]
        road_valid = (road[..., 6] > 0) & torch.isfinite(road[..., :7]).all(dim=-1)
        road_distance = torch.linalg.norm(road[..., :2], dim=-1)
        road_valid &= road_distance <= self.config.observation_radius_m
        order = torch.argsort(torch.where(road_valid, road_distance, torch.inf), dim=-1)
        order = order[..., :self.config.num_road_segments]
        road = torch.gather(road, 2, order[..., None].expand(-1, -1, -1, road.shape[-1]))
        road = torch.where(torch.gather(road_valid, 2, order)[..., None], road, 0.0)
        controlled_mask = torch.zeros_like(self._scene_valid)
        if self.config.control_all_vehicles:
            controlled_mask = self._controlled_valid
        else:
            controlled_mask[:, :self.config.num_controlled] = True
        return _build_scene_graph_batch_jit(
            _to_jax(_finite_torch(vehicles)), _to_jax(self._scene_valid), _to_jax(controlled_mask),
            _to_jax(_finite_torch(road)), _to_jax(_finite_torch(speeds)),
            config=self.config,
        )

    def continuous_cost_batch(self):
        """Compute vehicle and RoadEdge costs in one batched GPU operation."""
        if self._kept_slots is None:
            raise RuntimeError("prepare_scenes must be called before requesting costs")
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        lidar_all = self.env.sim.lidar_tensor().to_torch()
        worlds = torch.arange(self.num_worlds, device=absolute.device)[:, None]
        vehicles = absolute[worlds, self._kept_slots]
        ego = vehicles[:, :self.config.num_controlled]
        all_indices = torch.arange(self.config.max_scene_vehicles, device=absolute.device)
        ego_indices = torch.arange(self.config.num_controlled, device=absolute.device)
        other_indices = torch.stack([all_indices[all_indices != idx] for idx in ego_indices])
        others = vehicles[:, other_indices]
        scale = float(getattr(self.env.config, "agent_size_scale", 1.0))
        other_valid = self._scene_valid[:, other_indices]
        cost, clearance = _continuous_collision_cost_batch_jit(
            _to_jax(ego[:, :, :2]), _to_jax(ego[:, :, 7]),
            _to_jax(ego[:, :, 10] * scale), _to_jax(ego[:, :, 11] * scale),
            _to_jax(others[:, :, :, :2]), _to_jax(others[:, :, :, 7]),
            _to_jax(others[:, :, :, 10] * scale), _to_jax(others[:, :, :, 11] * scale),
            _to_jax(other_valid), self.config.safety_margin_m, self.config.geometry,
        )
        road_cost, road_clearance = _road_edge_collision_cost_batch_jit(
            _to_jax(_finite_torch(lidar_all[worlds, self._controlled_slots(), 1])),
            _to_jax(ego[:, :, 10] * scale), _to_jax(ego[:, :, 11] * scale),
            self.config.safety_margin_m, self.config.geometry,
        )
        combined_cost = jnp.concatenate([cost, road_cost], axis=-1)
        min_clearance = jnp.minimum(clearance, road_clearance)
        return jnp.nan_to_num(combined_cost, nan=1.0, posinf=1.0, neginf=-1.0), jnp.nan_to_num(
            min_clearance, nan=-self.config.safety_margin_m,
            posinf=self.config.observation_radius_m,
            neginf=-self.config.observation_radius_m,
        )

    def reset_worlds(self, world_indices):
        """Reset prepared worlds and restore the current controlled selection."""
        indices = [int(value) for value in world_indices]
        if not indices:
            return
        self.env.sim.reset(indices)
        local = self.env.sim.self_observation_tensor().to_torch()
        controlled_state = self.env.sim.controlled_state_tensor().to_torch()
        response_type = self.env.sim.response_type_tensor().to_torch()
        for world in indices:
            controlled_state[world].zero_()
            ids = local[world, :, 7].round().to(torch.int64)
            valid = local[world, :, 1] > 0
            for agent_id in self.prepared_worlds[world].controlled_ids:
                match = torch.where(valid & (ids == agent_id))[0]
                if match.numel() != 1:
                    raise RuntimeError(f"cannot restore controlled ID {agent_id} in world {world}")
                controlled_state[world, match[0], 0] = 1
                response_type[world, match[0]] = 0  # Dynamic
            if self.config.background_behavior == "static":
                for agent_id in self.prepared_worlds[world].background_ids:
                    match = torch.where(valid & (ids == agent_id))[0]
                    if match.numel() != 1:
                        raise RuntimeError(
                            f"cannot restore background ID {agent_id} in world {world}"
                        )
                    response_type[world, match[0]] = 2  # Static
        self.env.cont_agent_mask = self.env.get_controlled_agents_mask()

    def local_graphs(self) -> list[EgoGraphBatch]:
        """Compatibility wrapper returning one five-root graph per world."""
        batch = self.local_graph_batch()
        return [jax.tree_util.tree_map(
            lambda value: None if value is None else value[world], batch,
            is_leaf=lambda value: value is None,
        ) for world in range(self.num_worlds)]

        # Kept below only as historical context; the vectorized path above is
        # the single implementation used at runtime.
        if not self.prepared_worlds:
            raise RuntimeError("prepare_scenes must be called before requesting graphs")
        ego_all = self.env.sim.self_observation_tensor().to_torch()
        partner_all = self.env.sim.partner_observations_tensor().to_torch()
        lidar_all = self.env.sim.lidar_tensor().to_torch()
        slots = self._controlled_slots()
        outputs = []

        for world, selection in enumerate(self.prepared_worlds):
            ego = ego_all[world, slots[world]].clone()
            ego[:, 1:4] *= float(getattr(self.env.config, "agent_size_scale", 1.0))
            selected_partner = torch.zeros(
                (self.config.num_controlled, self.config.max_neighbors, 9),
                dtype=partner_all.dtype, device=partner_all.device,
            )
            partner_valid = torch.zeros(
                (self.config.num_controlled, self.config.max_neighbors),
                dtype=torch.bool, device=partner_all.device,
            )
            partner_controlled = torch.zeros_like(partner_valid)
            controlled_ids = set(selection.controlled_ids)
            kept_ids = set(selection.kept_ids)

            for ego_offset, slot in enumerate(slots[world].tolist()):
                candidates = partner_all[world, slot]
                write = 0
                for candidate in candidates:
                    candidate_id = int(round(float(candidate[8].item())))
                    if candidate_id not in kept_ids or candidate_id == int(round(float(ego[ego_offset, 7].item()))):
                        continue
                    selected_partner[ego_offset, write] = candidate
                    selected_partner[ego_offset, write, 4:7] *= float(
                        getattr(self.env.config, "agent_size_scale", 1.0)
                    )
                    partner_valid[ego_offset, write] = True
                    partner_controlled[ego_offset, write] = candidate_id in controlled_ids
                    write += 1
                    if write == self.config.max_neighbors:
                        break
                # Vehicles outside GPUDrive's partner-observation radius stay
                # as masked slots. They remain in the physical scene and the
                # centralized critic, but are not invented in the local actor graph.

            lidar = lidar_all[world, slots[world]]
            outputs.append(_build_ego_graph_jit(
                _to_jax(ego), _to_jax(selected_partner), _to_jax(partner_valid),
                _to_jax(partner_controlled), _to_jax(lidar[:, 1]),
                _to_jax(lidar[:, 2]), config=self.config,
            ))
        return outputs

    def continuous_cost(self):
        """Compute per-agent vehicle/RoadEdge costs and minimum clearance."""
        return self.continuous_cost_batch()

        if not self.prepared_worlds:
            raise RuntimeError("prepare_scenes must be called before requesting costs")
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        lidar_all = self.env.sim.lidar_tensor().to_torch()
        costs, clearances = [], []
        slots = self._controlled_slots()
        scale = float(getattr(self.env.config, "agent_size_scale", 1.0))
        for world, selection in enumerate(self.prepared_worlds):
            ids = absolute[world, :, 13].round().to(torch.int64)
            valid = absolute[world, :, 10] > 0
            controlled_rows = []
            other_rows = []
            for controlled_id in selection.controlled_ids:
                row = torch.where(valid & (ids == controlled_id))[0]
                if row.numel() != 1:
                    raise RuntimeError(f"missing controlled vehicle ID {controlled_id} in world {world}")
                controlled_rows.append(row[0])
                neighbor_ids = [value for value in selection.kept_ids if value != controlled_id]
                neighbor_rows = []
                for neighbor_id in neighbor_ids:
                    neighbor = torch.where(valid & (ids == neighbor_id))[0]
                    if neighbor.numel() != 1:
                        raise RuntimeError(f"missing neighbor vehicle ID {neighbor_id} in world {world}")
                    neighbor_rows.append(neighbor[0])
                other_rows.append(torch.stack(neighbor_rows))
            controlled_rows = torch.stack(controlled_rows)
            other_rows = torch.stack(other_rows)
            ego = absolute[world, controlled_rows]
            others = absolute[world, other_rows]
            cost, clearance = _continuous_collision_cost_jit(
                ego_position=_to_jax(ego[:, :2]),
                ego_heading=_to_jax(ego[:, 7]),
                ego_length=_to_jax(ego[:, 10] * scale),
                ego_width=_to_jax(ego[:, 11] * scale),
                other_position=_to_jax(others[:, :, :2]),
                other_heading=_to_jax(others[:, :, 7]),
                other_length=_to_jax(others[:, :, 10] * scale),
                other_width=_to_jax(others[:, :, 11] * scale),
                valid_mask=jnp.ones((self.config.num_controlled, self.config.max_neighbors), dtype=bool),
                margin_m=self.config.safety_margin_m,
                mode=self.config.geometry,
            )
            road_cost, road_clearance = _road_edge_collision_cost_jit(
                road_edge_samples=_to_jax(lidar_all[world, slots[world], 1]),
                ego_length=_to_jax(ego[:, 10] * scale),
                ego_width=_to_jax(ego[:, 11] * scale),
                margin_m=self.config.safety_margin_m,
                mode=self.config.geometry,
            )
            costs.append(jnp.concatenate([cost, road_cost], axis=-1))
            clearances.append(jnp.minimum(clearance, road_clearance))
        return jnp.stack(costs), jnp.stack(clearances)

    def collision_categories(self, info_tensor=None):
        """Classify native vehicle collisions by the nearest collision partner."""
        info = self.env.sim.info_tensor().to_torch() if info_tensor is None else info_tensor
        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        slots = self._controlled_slots()
        cc = torch.zeros((self.num_worlds, self.config.num_controlled), dtype=torch.bool, device=absolute.device)
        cb = torch.zeros_like(cc)
        for world, selection in enumerate(self.prepared_worlds):
            ids = absolute[world, :, 13].round().to(torch.int64)
            valid = absolute[world, :, 10] > 0
            controlled_ids = set(selection.controlled_ids)
            for offset, slot in enumerate(slots[world].tolist()):
                if self._controlled_valid is not None and not bool(
                    self._controlled_valid[world, offset]
                ):
                    continue
                native_collision = bool(info[world, slot, 1:3].sum() > 0)
                if not native_collision:
                    continue
                ego_pos = absolute[world, slot, :2]
                candidates = []
                for neighbor_id in selection.kept_ids:
                    if neighbor_id == int(ids[slot]):
                        continue
                    row = torch.where(valid & (ids == neighbor_id))[0]
                    if row.numel() == 1:
                        candidates.append((float(torch.linalg.norm(absolute[world, row[0], :2] - ego_pos)), neighbor_id))
                if candidates:
                    partner_id = min(candidates)[1]
                    cc[world, offset] = partner_id in controlled_ids
                    cb[world, offset] = partner_id not in controlled_ids
        return cc, cb

    def scene_graphs(self):
        """Return one privileged centralized Vl graph per world."""
        batch = self.scene_graph_batch()
        return [jax.tree_util.tree_map(
            lambda value: None if value is None else value[world], batch,
            is_leaf=lambda value: value is None,
        ) for world in range(self.num_worlds)]

        absolute = self.env.sim.absolute_self_observation_tensor().to_torch()
        local = self.env.sim.self_observation_tensor().to_torch()
        roadmap = self.env.sim.agent_roadmap_tensor().to_torch()
        slots = self._controlled_slots()
        outputs = []
        for world, selection in enumerate(self.prepared_worlds):
            ids = absolute[world, :, 13].round().to(torch.int64)
            valid = absolute[world, :, 10] > 0
            ordered_ids = selection.controlled_ids + selection.background_ids
            rows = []
            for agent_id in ordered_ids:
                match = torch.where(valid & (ids == agent_id))[0]
                if match.numel() != 1:
                    raise RuntimeError(f"missing scene vehicle ID {agent_id}")
                rows.append(match[0])
            vehicles = absolute[world, torch.stack(rows)].clone()
            speeds = local[world, torch.stack(rows), 0].clone()
            vehicles[:, 10:13] *= float(getattr(self.env.config, "agent_size_scale", 1.0))
            road = roadmap[world, slots[world, 0]].clone()
            outputs.append(_build_scene_graph_jit(
                _to_jax(vehicles),
                jnp.arange(self.config.num_vehicles) < self.config.num_controlled,
                _to_jax(road), _to_jax(speeds), self.config,
            ))
        return outputs

    def step(self, normalized_actions: Array):
        """Execute normalized (acceleration, steering) actions for live controlled cars."""
        actions = torch.utils.dlpack.from_dlpack(normalized_actions)
        expected = (self.num_worlds, self.config.num_controlled, 2)
        if actions.shape != expected:
            raise ValueError(f"expected normalized actions {expected}, got {actions.shape}")
        full = torch.zeros(
            (self.num_worlds, self.env.max_agent_count, 3),
            dtype=torch.float32, device=self.env.device,
        )
        slots = self._controlled_slots()
        action_t = actions.to(device=self.env.device, dtype=torch.float32)
        accel_limit = float(torch.as_tensor(self.env.config.accel_actions).abs().max())
        steering_limit = float(torch.as_tensor(self.env.config.steer_actions).abs().max())
        for world in range(self.num_worlds):
            valid = self._controlled_valid[world]
            valid_slots = slots[world, valid]
            full[world, valid_slots, 0] = (
                torch.clamp(action_t[world, valid, 0], -1.0, 1.0) * accel_limit
            )
            full[world, valid_slots, 1] = (
                torch.clamp(action_t[world, valid, 1], -1.0, 1.0) * steering_limit
            )
        self.env.step_dynamics(full)
        return self.env.get_rewards(), self.env.get_dones(), self.env.sim.info_tensor().to_torch().clone()
