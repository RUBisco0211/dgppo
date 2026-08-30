"""Replay storage for Deep-QP safety transitions."""

import pickle
from pathlib import Path

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .data import SafetyBatch


class SafetyReplayBuffer:
    """A bounded host-memory replay buffer storing full local graph transitions."""

    def __init__(self, size: int, seed: int = 0):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        self._buffer: SafetyBatch | None = None
        self._capacity = 0
        self._length = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    def append(self, batch: SafetyBatch) -> None:
        batch_np = jtu.tree_map(np.asarray, batch)
        n_new = min(len(batch_np.actions), self.size)
        if len(batch_np.actions) > n_new:
            batch_np = jtu.tree_map(lambda x: x[-n_new:], batch_np)
        required = min(self.size, self._length + n_new)
        if self._buffer is None:
            self._capacity = min(self.size, max(1024, required))
            self._buffer = jtu.tree_map(
                lambda x: np.empty((self._capacity,) + x.shape[1:], dtype=x.dtype), batch_np
            )
        elif required > self._capacity:
            new_capacity = min(self.size, max(required, self._capacity * 2))

            def grow(storage):
                enlarged = np.empty((new_capacity,) + storage.shape[1:], dtype=storage.dtype)
                enlarged[:self._length] = storage[:self._length]
                return enlarged

            self._buffer = jtu.tree_map(grow, self._buffer)
            self._capacity = new_capacity
            self._cursor = self._length

        indices = (np.arange(n_new) + self._cursor) % self._capacity

        def write(storage, values):
            storage[indices] = values
            return storage

        self._buffer = jtu.tree_map(write, self._buffer, batch_np)
        self._cursor = int((self._cursor + n_new) % self._capacity)
        self._length = min(self.size, self._length + n_new)

    def sample(
            self,
            batch_size: int,
            boundary_fraction: float = 0.5,
            boundary_width: float = 0.05,
    ) -> SafetyBatch:
        if self._buffer is None or self.length < batch_size:
            raise ValueError(f"need {batch_size} transitions, buffer contains {self.length}")
        if not 0.0 <= boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must be in [0, 1]")
        if boundary_width < 0.0:
            raise ValueError("boundary_width must be non-negative")
        n_boundary = int(round(batch_size * boundary_fraction))
        candidate_size = min(self.length, max(batch_size * 8, batch_size))
        candidates = self._rng.integers(0, self.length, size=candidate_size)
        candidate_constraints = self._buffer.constraints[candidates]
        near_boundary = np.min(np.abs(candidate_constraints), axis=-1) <= boundary_width
        unsafe = np.min(candidate_constraints, axis=-1) < 0.0
        priority_candidates = candidates[near_boundary | unsafe]
        n_priority = min(n_boundary, len(priority_candidates))
        if n_priority > 0:
            priority_indices = self._rng.choice(priority_candidates, size=n_priority, replace=True)
        else:
            priority_indices = np.empty((0,), dtype=np.int64)
        uniform_indices = self._rng.integers(0, self.length, size=batch_size - n_priority)
        indices = np.concatenate([priority_indices, uniform_indices])
        self._rng.shuffle(indices)
        return jtu.tree_map(lambda x: jnp.asarray(x[indices]), self._buffer)

    @property
    def length(self) -> int:
        return self._length

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as file:
            pickle.dump({
                "size": self.size,
                "buffer": self._buffer,
                "capacity": self._capacity,
                "length": self._length,
                "cursor": self._cursor,
                "rng_state": self._rng.bit_generator.state,
            }, file)

    def load(self, path: str | Path, expected_n_agents: int | None = None) -> None:
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        self.size = int(payload["size"])
        self._buffer = payload["buffer"]
        if (
            expected_n_agents is not None
            and self._buffer is not None
            and self._buffer.actions.shape[-2] != expected_n_agents
        ):
            raise ValueError("safety replay n_agents does not match the current filter")
        self._length = int(payload.get("length", 0 if self._buffer is None else len(self._buffer.actions)))
        self._capacity = int(payload.get(
            "capacity", 0 if self._buffer is None else len(self._buffer.actions)
        ))
        self._cursor = int(payload.get(
            "cursor", 0 if self._capacity == 0 else self._length % self._capacity
        ))
        if "rng_state" in payload:
            self._rng.bit_generator.state = payload["rng_state"]
