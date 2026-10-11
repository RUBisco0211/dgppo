"""Deterministic scene reduction and controlled-agent selection helpers."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ...utils.typing import Array


class VehicleSelection(NamedTuple):
    kept_indices: Array
    controlled_indices: Array
    background_indices: Array


def select_interacting_vehicles(
    positions: Array,
    valid_mask: Array,
    key: Array,
    num_controlled: int = 3,
    num_background: int = 5,
) -> VehicleSelection:
    """Select a random anchor, its nearest vehicles, and control three.

    The anchor and its two closest neighbors become controlled vehicles.  The
    remaining closest vehicles become background traffic. Invalid slots are
    assigned infinite distance and can never be selected.
    """
    total = num_controlled + num_background
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (n_vehicle_slots, 2)")
    if valid_mask.shape != positions.shape[:1]:
        raise ValueError("valid_mask must match the vehicle-slot dimension")
    if positions.shape[0] < total:
        raise ValueError(f"at least {total} vehicle slots are required")

    valid_count = jnp.sum(valid_mask)
    # The result remains shape-static under jit. Callers should reject a scene
    # when this flag is false before using the selected IDs.
    logits = jnp.where(valid_mask, 0.0, -jnp.inf)
    anchor = jax.random.categorical(key, logits).astype(jnp.int32)
    distances = jnp.linalg.norm(positions - positions[anchor], axis=-1)
    distances = jnp.where(valid_mask, distances, jnp.inf)
    distances = distances.at[anchor].set(-1.0)
    kept = jnp.argsort(distances)[:total]
    controlled = kept[:num_controlled]
    background = kept[num_controlled:]
    invalid_index = jnp.asarray(-1, dtype=jnp.int32)
    kept = jnp.where(valid_count >= total, kept, invalid_index)
    controlled = jnp.where(valid_count >= total, controlled, invalid_index)
    background = jnp.where(valid_count >= total, background, invalid_index)
    return VehicleSelection(kept, controlled, background)
