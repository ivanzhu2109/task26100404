"""GPU gang scheduler: strict queue-order optimal, retention-aware."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from ._flow import max_bipartite_matching, min_cost_assignment
from .models import Job, Plan


class GpuScheduler:
    """Registers candidate jobs and produces GPU assignment plans."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._next_seq = 0
        self._last_plan: Plan | None = None

    def register(
        self,
        job_id: str,
        priority: int,
        gpu_count: int,
        compatible_gpus: Iterable[int],
    ) -> None:
        """Register (or replace) a candidate job."""
        if not 1 <= priority <= 5:
            raise ValueError("priority must be in 1..5")
        if gpu_count < 1:
            raise ValueError("gpu_count must be >= 1")
        compatible = frozenset(compatible_gpus)
        if len(compatible) < gpu_count:
            raise ValueError("compatible_gpus smaller than gpu_count")
        existing = self._jobs.get(job_id)
        seq = existing.seq if existing is not None else self._next_seq
        self._next_seq = max(self._next_seq, seq + 1)
        self._jobs[job_id] = Job(
            job_id=job_id,
            priority=priority,
            gpu_count=gpu_count,
            compatible_gpus=compatible,
            seq=seq,
        )

    def withdraw(self, job_id: str) -> None:
        """Remove a candidate job; missing ids are ignored."""
        self._jobs.pop(job_id, None)

    def plan(self, available_gpus: Iterable[int]) -> Plan:
        """Compute the queue-order-optimal plan over available GPUs."""
        available = frozenset(available_gpus)
        queue = sorted(self._jobs.values(), key=lambda job: (job.priority, job.seq))

        gpu_ids = sorted(available)
        gpu_index = {gpu: idx for idx, gpu in enumerate(gpu_ids)}

        def edges_of(job: Job) -> list[int]:
            return [
                gpu_index[gpu]
                for gpu in sorted(job.compatible_gpus & available)
            ]

        # Phase 1: lexicographically maximal inclusion set in queue order.
        selected: list[Job] = []
        for job in queue:
            candidate = selected + [job]
            demands = [item.gpu_count for item in candidate]
            edges = [edges_of(item) for item in candidate]
            if max_bipartite_matching(demands, edges, len(gpu_ids)):
                selected.append(job)

        # Phase 2: among feasible assignments of the selected set, keep as
        # many (job, gpu) pairs from the previous plan as possible.
        previous = self._last_plan
        demands = [job.gpu_count for job in selected]
        edges = [edges_of(job) for job in selected]
        costs: list[dict[int, int]] = []
        for job, job_edges in zip(selected, edges):
            kept = previous.gpu_ids_for(job.job_id) if previous else frozenset()
            costs.append(
                {
                    idx: -1 if gpu_ids[idx] in kept else 0
                    for idx in job_edges
                }
            )
        chosen = min_cost_assignment(demands, edges, costs, len(gpu_ids))
        assert chosen is not None  # feasibility already proven in phase 1

        assignments = {
            job.job_id: frozenset(gpu_ids[idx] for idx in chosen[i])
            for i, job in enumerate(selected)
        }
        plan = Plan(MappingProxyType(assignments))
        self._last_plan = plan
        return plan
