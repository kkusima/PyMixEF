"""Deterministic counter-based random streams for parallel scientific workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _label_words(label: str) -> tuple[int, ...]:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=16).digest()
    return tuple(
        int.from_bytes(digest[position : position + 4], "little")
        for position in range(0, len(digest), 4)
    )


@dataclass(frozen=True, slots=True)
class RandomStreamManager:
    """Create order-independent NumPy Philox streams from a recorded root seed."""

    seed: int
    namespace: str = "pymixef"

    def generator(
        self,
        component: str,
        *,
        replicate: int = 0,
        chain: int = 0,
    ) -> np.random.Generator:
        if replicate < 0 or chain < 0:
            raise ValueError("replicate and chain identifiers must be nonnegative.")
        spawn_key = (
            *_label_words(self.namespace),
            *_label_words(component),
            int(replicate),
            int(chain),
        )
        sequence = np.random.SeedSequence(int(self.seed), spawn_key=spawn_key)
        return np.random.Generator(np.random.Philox(sequence))

    def replicates(self, component: str, count: int) -> Iterable[np.random.Generator]:
        if count < 0:
            raise ValueError("count must be nonnegative.")
        for replicate in range(count):
            yield self.generator(component, replicate=replicate)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": "numpy.Philox",
            "seed": int(self.seed),
            "namespace": self.namespace,
            "stream_derivation": "SeedSequence spawn key from BLAKE2b labels",
        }


def random_streams(seed: int, *, namespace: str = "pymixef") -> RandomStreamManager:
    """Construct a named counter-based stream manager."""

    return RandomStreamManager(int(seed), namespace)
