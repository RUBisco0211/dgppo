"""Host-memory replay storage for GCBF+ graph transitions."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .data import GCBFTransitionBatch


class GCBFReplayBuffer:
    def __init__(self, size: int, seed: int = 0):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        self._buffer: GCBFTransitionBatch | None = None
        self._capacity = 0
        self._length = 0
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    @property
    def length(self) -> int:
        return self._length

    def append(self, batch: GCBFTransitionBatch) -> None:
        batch = jtu.tree_map(np.asarray, batch)
        n_new = min(len(batch.safe_mask), self.size)
        if len(batch.safe_mask) > n_new:
            batch = jtu.tree_map(lambda value: value[-n_new:], batch)
        required = min(self.size, self._length + n_new)
        if self._buffer is None:
            self._capacity = min(self.size, max(1024, required))
            self._buffer = jtu.tree_map(
                lambda value: np.empty(
                    (self._capacity,) + value.shape[1:], dtype=value.dtype
                ),
                batch,
            )
        elif required > self._capacity:
            new_capacity = min(self.size, max(required, self._capacity * 2))

            def grow(value):
                enlarged = np.empty(
                    (new_capacity,) + value.shape[1:], dtype=value.dtype
                )
                enlarged[: self._length] = value[: self._length]
                return enlarged

            self._buffer = jtu.tree_map(grow, self._buffer)
            self._capacity = new_capacity
            self._cursor = self._length

        indices = (np.arange(n_new) + self._cursor) % self._capacity

        def write(storage, values):
            storage[indices] = values
            return storage

        self._buffer = jtu.tree_map(write, self._buffer, batch)
        self._cursor = int((self._cursor + n_new) % self._capacity)
        self._length = required

    def sample(self, batch_size: int, unsafe_fraction: float = 0.5) -> GCBFTransitionBatch:
        if self._buffer is None or self.length < batch_size:
            raise ValueError(f"need {batch_size} transitions, buffer contains {self.length}")
        if not 0.0 <= unsafe_fraction <= 1.0:
            raise ValueError("unsafe_fraction must be in [0, 1]")

        # The backing arrays are deliberately over-allocated.  Never inspect
        # the unwritten tail: ``np.empty`` may contain arbitrary bits which can
        # look like unsafe labels and make prioritized sampling return an
        # uninitialized graph.
        unsafe_rows = np.flatnonzero(
            np.asarray(self._buffer.unsafe_mask[: self.length]).any(axis=-1)
        )
        requested_unsafe = int(round(batch_size * unsafe_fraction))
        n_unsafe = requested_unsafe if len(unsafe_rows) else 0
        if n_unsafe:
            selected_unsafe = self._rng.choice(unsafe_rows, size=n_unsafe, replace=True)
        else:
            selected_unsafe = np.empty((0,), dtype=np.int64)
        selected_other = self._rng.integers(0, self.length, size=batch_size - n_unsafe)
        indices = np.concatenate([selected_unsafe, selected_other])
        self._rng.shuffle(indices)
        return jtu.tree_map(lambda value: jnp.asarray(value[indices]), self._buffer)

    def save(self, path: str | Path) -> None:
        compact = None
        if self._buffer is not None:
            compact = jtu.tree_map(
                lambda value: np.array(value[: self._length], copy=True), self._buffer
            )
        with Path(path).open("wb") as file:
            pickle.dump(
                {
                    "size": self.size,
                    "buffer": compact,
                    "capacity": self._length,
                    "length": self._length,
                    "cursor": 0 if self._length == self.size else self._length,
                    "rng_state": self._rng.bit_generator.state,
                },
                file,
            )

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        self.size = int(payload["size"])
        self._buffer = payload["buffer"]
        self._length = int(payload["length"])
        self._capacity = int(
            payload.get("capacity", 0 if self._buffer is None else len(self._buffer.safe_mask))
        )
        self._cursor = int(
            payload.get(
                "cursor", 0 if self._capacity == 0 else self._length % self._capacity
            )
        )
        self._rng.bit_generator.state = payload["rng_state"]
