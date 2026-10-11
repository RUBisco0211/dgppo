"""Continuous vehicle-clearance models used by the GPUDrive DGPPO adapter."""

from typing import Literal, Tuple

import jax.numpy as jnp

from ...utils.typing import Array


def _axes(heading: Array) -> Array:
    forward = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1)
    lateral = jnp.stack([-jnp.sin(heading), jnp.cos(heading)], axis=-1)
    return jnp.stack([forward, lateral], axis=-2)


def oriented_box_signed_clearance(
    ego_position: Array,
    ego_heading: Array,
    ego_length: Array,
    ego_width: Array,
    other_position: Array,
    other_heading: Array,
    other_length: Array,
    other_width: Array,
) -> Array:
    """Return a differentiable SAT-style signed separation for two OBBs.

    Positive values indicate a separating axis, zero is contact, and negative
    values indicate overlap.  For diagonally separated boxes this is a
    conservative clearance rather than the exact polygon-to-polygon distance;
    that behavior is desirable for a safety cost.
    """
    axes_ego = _axes(ego_heading)
    axes_other = _axes(other_heading)
    broadcast_shape = jnp.broadcast_shapes(axes_ego.shape[:-2], axes_other.shape[:-2])
    axes_ego = jnp.broadcast_to(axes_ego, broadcast_shape + (2, 2))
    axes_other = jnp.broadcast_to(axes_other, broadcast_shape + (2, 2))
    axes = jnp.concatenate([axes_ego, axes_other], axis=-2)
    delta = jnp.broadcast_to(other_position - ego_position, broadcast_shape + (2,))

    ego_half = jnp.stack([ego_length, ego_width], axis=-1) * 0.5
    other_half = jnp.stack([other_length, other_width], axis=-1) * 0.5
    ego_half = jnp.broadcast_to(ego_half, broadcast_shape + (2,))
    other_half = jnp.broadcast_to(other_half, broadcast_shape + (2,))

    ego_projection = jnp.abs(jnp.einsum("...ad,...kd->...ak", axes, axes_ego))
    other_projection = jnp.abs(jnp.einsum("...ad,...kd->...ak", axes, axes_other))
    ego_radius = jnp.sum(ego_projection * ego_half[..., None, :], axis=-1)
    other_radius = jnp.sum(other_projection * other_half[..., None, :], axis=-1)
    center_projection = jnp.abs(jnp.einsum("...ad,...d->...a", axes, delta))
    axis_gaps = center_projection - ego_radius - other_radius
    return jnp.max(axis_gaps, axis=-1)


def ellipse_signed_clearance(
    ego_position: Array,
    ego_heading: Array,
    ego_length: Array,
    ego_width: Array,
    other_position: Array,
    other_heading: Array,
    other_length: Array,
    other_width: Array,
) -> Array:
    """Approximate signed clearance using directional ellipse support radii."""
    delta = other_position - ego_position
    center_distance = jnp.linalg.norm(delta, axis=-1)
    direction = delta / jnp.maximum(center_distance[..., None], 1e-6)

    def support_radius(heading, length, width, direction_):
        local_x = direction_[..., 0] * jnp.cos(heading) + direction_[..., 1] * jnp.sin(heading)
        local_y = -direction_[..., 0] * jnp.sin(heading) + direction_[..., 1] * jnp.cos(heading)
        return jnp.sqrt(jnp.square(0.5 * length * local_x) + jnp.square(0.5 * width * local_y))

    ego_radius = support_radius(ego_heading, ego_length, ego_width, direction)
    other_radius = support_radius(other_heading, other_length, other_width, -direction)
    return center_distance - ego_radius - other_radius


def pairwise_signed_clearance(
    ego_position: Array,
    ego_heading: Array,
    ego_length: Array,
    ego_width: Array,
    other_position: Array,
    other_heading: Array,
    other_length: Array,
    other_width: Array,
    mode: Literal["oriented_box", "ellipse"] = "oriented_box",
) -> Array:
    if mode == "oriented_box":
        return oriented_box_signed_clearance(
            ego_position, ego_heading, ego_length, ego_width,
            other_position, other_heading, other_length, other_width,
        )
    if mode == "ellipse":
        return ellipse_signed_clearance(
            ego_position, ego_heading, ego_length, ego_width,
            other_position, other_heading, other_length, other_width,
        )
    raise ValueError(f"unknown geometry mode: {mode}")


def continuous_collision_cost(
    ego_position: Array,
    ego_heading: Array,
    ego_length: Array,
    ego_width: Array,
    other_position: Array,
    other_heading: Array,
    other_length: Array,
    other_width: Array,
    valid_mask: Array,
    margin_m: float = 0.5,
    mode: Literal["oriented_box", "ellipse"] = "oriented_box",
) -> Tuple[Array, Array]:
    """Return worst-neighbor cost and the corresponding minimum clearance."""
    clearance = pairwise_signed_clearance(
        ego_position[..., None, :], ego_heading[..., None],
        ego_length[..., None], ego_width[..., None], other_position,
        other_heading, other_length, other_width, mode=mode,
    )
    clearance = jnp.where(valid_mask, clearance, jnp.inf)
    min_clearance = jnp.min(clearance, axis=-1)
    min_clearance = jnp.where(jnp.any(valid_mask, axis=-1), min_clearance, jnp.inf)
    # Match the bounded continuous-cost convention used by the original
    # DGPPO navigation/LiDAR environments. The zero level set remains exactly
    # at ``margin_m`` and positive values still mean unsafe.
    cost = jnp.clip(margin_m - min_clearance, a_min=-1.0, a_max=1.0)
    return cost[..., None], min_clearance


def road_edge_collision_cost(
    road_edge_samples: Array,
    ego_length: Array,
    ego_width: Array,
    margin_m: float = 0.5,
    mode: Literal["oriented_box", "ellipse"] = "oriented_box",
) -> Tuple[Array, Array]:
    """Continuous ego-to-RoadEdge cost from ego-centric LiDAR hits.

    LiDAR hit positions are expressed as ``(right, forward)`` from the ego
    center. The vehicle support radius is subtracted from the ray depth so the
    returned clearance is measured from the vehicle body, not its center.
    RoadLine hits are intentionally excluded: they describe lane structure,
    while RoadEdge is the obstacle counterpart used by DGPPO's road cost.
    """
    depth = road_edge_samples[..., 0]
    encoded_type = road_edge_samples[..., 1].round().astype(jnp.int32)
    hit = road_edge_samples[..., 2:4]
    valid = (depth > 0.0) & (encoded_type == 1)
    direction = hit / jnp.maximum(depth[..., None], 1e-6)
    half_length = 0.5 * ego_length[..., None]
    half_width = 0.5 * ego_width[..., None]
    if mode == "oriented_box":
        support = jnp.abs(direction[..., 0]) * half_width + \
            jnp.abs(direction[..., 1]) * half_length
    elif mode == "ellipse":
        support = jnp.sqrt(
            jnp.square(direction[..., 0] * half_width)
            + jnp.square(direction[..., 1] * half_length)
        )
    else:
        raise ValueError(f"unknown geometry mode: {mode}")
    clearance = jnp.where(valid, depth - support, jnp.inf)
    min_clearance = jnp.min(clearance, axis=-1)
    min_clearance = jnp.where(jnp.any(valid, axis=-1), min_clearance, jnp.inf)
    cost = jnp.clip(margin_m - min_clearance, a_min=-1.0, a_max=1.0)
    return cost[..., None], min_clearance
