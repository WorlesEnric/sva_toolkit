"""Seedable random-number generation for SVA synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random
import secrets
from typing import TypeVar


T = TypeVar("T")


def resolve_seed(explicit: int | None) -> int:
    """Return an explicit seed or draw a fresh one from the OS."""
    if explicit is not None:
        return explicit
    return secrets.randbits(32)


class GenerationRng:
    """Small wrapper around ``random.Random`` for generator call sites."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed_value = resolve_seed(seed)
        self._random = Random(self.seed_value)

    def random(self) -> float:
        return self._random.random()

    def randint(self, a: int, b: int) -> int:
        return self._random.randint(a, b)

    def choice(self, seq: Sequence[T]) -> T:
        return self._random.choice(seq)

    def choices(
        self,
        population: Sequence[T],
        weights: Sequence[float] | None = None,
        *,
        cum_weights: Sequence[float] | None = None,
        k: int = 1,
    ) -> list[T]:
        return self._random.choices(population, weights=weights, cum_weights=cum_weights, k=k)

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        return self._random.sample(population, k)

    def shuffle(self, x: list[T]) -> None:
        self._random.shuffle(x)

    def uniform(self, a: float, b: float) -> float:
        return self._random.uniform(a, b)

    def seed(self, a: int | None = None) -> None:
        self.seed_value = resolve_seed(a)
        self._random.seed(self.seed_value)
