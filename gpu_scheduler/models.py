"""Type definitions for the GPU gang scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class Job:
    """A candidate job registered by the caller.

    Attributes are immutable for the lifetime of the registration.
    """

    job_id: str
    priority: int  # 1 (highest) .. 5 (lowest)
    gpu_count: int  # number of distinct GPUs required
    compatible_gpus: frozenset[int]  # GPU ids the job may run on
    seq: int = field(default=0, compare=False)  # registration order


@dataclass(frozen=True)
class Plan:
    """Result of one scheduling round: job_id -> assigned GPU ids."""

    assignments: Mapping[str, frozenset[int]]

    def gpu_ids_for(self, job_id: str) -> frozenset[int]:
        return self.assignments.get(job_id, frozenset())


EMPTY_PLAN = Plan(MappingProxyType({}))
