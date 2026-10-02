"""GPU 整组排程器：登记/撤回候选作业，生成最优分配计划。"""
from __future__ import annotations

from typing import Dict, FrozenSet, List

from .matching import best_assignment, feasible
from .models import Job, Plan


class Scheduler:
    """维护候选作业队列与上一份计划，按需生成新计划。"""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []  # 登记顺序，同级优先级按此排序
        self._previous: Dict[str, FrozenSet[str]] = {}

    def register(self, job: Job) -> None:
        """登记候选作业。job_id 必须唯一。"""
        if job.job_id in self._jobs:
            raise ValueError(f"作业已登记: {job.job_id}")
        self._jobs[job.job_id] = job
        self._order.append(job.job_id)

    def withdraw(self, job_id: str) -> None:
        """撤回候选作业；其旧分配随之失效。"""
        if job_id not in self._jobs:
            raise KeyError(f"作业不存在: {job_id}")
        del self._jobs[job_id]
        self._order.remove(job_id)
        self._previous.pop(job_id, None)

    def _queue(self) -> List[Job]:
        """按 (优先级, 登记先后) 排列的候选队列。"""
        rank = {job_id: i for i, job_id in enumerate(self._order)}
        return sorted(self._jobs.values(), key=lambda j: (j.priority, rank[j.job_id]))

    def plan(self, available_gpus: FrozenSet[str]) -> Plan:
        """基于当前可用 GPU 集合生成最优计划，并记为下一次的上一份计划。

        1. 按队列顺序贪心构造字典序最大的入选集合：逐个尝试加入，
           加入后整体仍可行才保留（无法容纳的作业跳过，继续看后面的）。
        2. 入选集合固定后，在所有可行分配中选保留上一份配对最多的一份。
        """
        available = frozenset(available_gpus)
        selected: List[Job] = []
        for job in self._queue():
            trial = selected + [job]
            if feasible(trial, available):
                selected = trial
        assignments = best_assignment(selected, available, self._previous)
        self._previous = assignments
        return Plan(assignments)
