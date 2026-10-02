"""数据类型定义：候选作业与分配计划。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class Job:
    """候选推理作业。属性在登记后不变。"""

    job_id: str
    priority: int  # 1（最高）至 5（最低）
    gpu_count: int  # 所需 GPU 数量，>= 1
    compatible_gpus: FrozenSet[str]  # 兼容 GPU 编号集合

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 5:
            raise ValueError("priority 必须在 1 至 5 之间")
        if self.gpu_count < 1:
            raise ValueError("gpu_count 必须 >= 1")
        if not self.compatible_gpus:
            raise ValueError("compatible_gpus 不能为空")


@dataclass(frozen=True)
class Plan:
    """一次排程的输出：job_id -> 分到的 GPU 编号集合。"""

    assignments: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    def assigned_gpus(self) -> FrozenSet[str]:
        """计划中所有被占用的 GPU。"""
        result: set[str] = set()
        for gpus in self.assignments.values():
            result.update(gpus)
        return frozenset(result)
