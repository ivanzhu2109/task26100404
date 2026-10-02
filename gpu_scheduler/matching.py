"""内部匹配算法：可行性判定与保留旧配对的最优分配。"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

from .models import Job


def _expand_slots(selected: List[Job]) -> List[Tuple[int, str]]:
    """把每个作业展开为 gpu_count 个相同槽位，返回 (作业下标, GPU) 候选边左侧。"""
    slots: List[Tuple[int, str]] = []
    for idx, job in enumerate(selected):
        for _ in range(job.gpu_count):
            slots.append((idx, job.job_id))
    return slots


def feasible(selected: List[Job], available_gpus: FrozenSet[str]) -> bool:
    """判断所选作业集合能否在可用 GPU 内完成分配（不允许部分分配）。

    二分图匹配：左侧为作业槽位，右侧为可用 GPU，边表示兼容。
    所有槽位都能匹配到不同 GPU 时才可行。
    """
    total = sum(job.gpu_count for job in selected)
    if total > len(available_gpus):
        return False
    slots = _expand_slots(selected)
    # 每个槽位的候选 GPU（兼容且可用）
    candidates: List[List[str]] = []
    for idx, _ in slots:
        compat = selected[idx].compatible_gpus & available_gpus
        candidates.append(sorted(compat))

    match_of: Dict[str, int] = {}  # GPU -> 槽位下标

    def augment(slot: int, seen: set[str]) -> bool:
        for gpu in candidates[slot]:
            if gpu in seen:
                continue
            seen.add(gpu)
            if gpu not in match_of or augment(match_of[gpu], seen):
                match_of[gpu] = slot
                return True
        return False

    matched = 0
    for slot in range(len(slots)):
        if augment(slot, set()):
            matched += 1
        else:
            return False
    return matched == total


def best_assignment(
    selected: List[Job],
    available_gpus: FrozenSet[str],
    previous: Dict[str, FrozenSet[str]],
) -> Dict[str, FrozenSet[str]]:
    """在可行分配中，选择保留 previous 中 作业-GPU 配对 数量最多的一份。

    最小费用最大流：每张卡容量 1，旧配对边费用 0、其余边费用 1，
    满流时费用最小即保留配对最多。调用前需保证 feasible 为真。
    """
    slots = _expand_slots(selected)
    gpus = sorted(available_gpus)
    n_slots = len(slots)
    # 节点编号：0=源, 1..n_slots=槽位, 之后为 GPU, 最后为汇
    gpu_node = {gpu: 1 + n_slots + i for i, gpu in enumerate(gpus)}
    sink = 1 + n_slots + len(gpus)
    n_nodes = sink + 1

    # 邻接表存 [to, cap, cost, 反向边下标]
    graph: List[List[List[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(frm: int, to: int, cap: int, cost: int) -> None:
        graph[frm].append([to, cap, cost, len(graph[to])])
        graph[to].append([frm, 0, -cost, len(graph[frm]) - 1])

    for slot in range(n_slots):
        add_edge(0, 1 + slot, 1, 0)
    for slot, (idx, job_id) in enumerate(slots):
        kept = previous.get(job_id, frozenset())
        for gpu in sorted(selected[idx].compatible_gpus & available_gpus):
            cost = 0 if gpu in kept else 1
            add_edge(1 + slot, gpu_node[gpu], 1, cost)
    for gpu in gpus:
        add_edge(gpu_node[gpu], sink, 1, 0)

    # 最小费用最大流（SPFA 找最短路增广）
    flow = 0
    while flow < n_slots:
        dist = [float("inf")] * n_nodes
        inqueue = [False] * n_nodes
        prev_node = [-1] * n_nodes
        prev_edge = [-1] * n_nodes
        dist[0] = 0
        queue = [0]
        inqueue[0] = True
        while queue:
            node = queue.pop(0)
            inqueue[node] = False
            for ei, edge in enumerate(graph[node]):
                to, cap, cost, _ = edge
                if cap > 0 and dist[node] + cost < dist[to]:
                    dist[to] = dist[node] + cost
                    prev_node[to] = node
                    prev_edge[to] = ei
                    if not inqueue[to]:
                        queue.append(to)
                        inqueue[to] = True
        if dist[sink] == float("inf"):
            break  # 不应发生：调用前已确认可行
        node = sink
        while node != 0:
            edge = graph[prev_node[node]][prev_edge[node]]
            edge[1] -= 1
            graph[node][edge[3]][1] += 1
            node = prev_node[node]
        flow += 1

    # 从流结果还原 作业 -> GPU 集合
    result: Dict[str, set[str]] = {}
    for slot, (_, job_id) in enumerate(slots):
        for edge in graph[1 + slot]:
            to, cap, _, _ = edge
            if to != 0 and cap == 0:  # 槽位 -> GPU 的边被用掉
                gpu = gpus[to - 1 - n_slots]
                result.setdefault(job_id, set()).add(gpu)
    return {job_id: frozenset(gpus) for job_id, gpus in result.items()}
