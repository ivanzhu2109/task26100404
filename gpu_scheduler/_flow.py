"""Internal flow-network helpers (max flow / min-cost max flow)."""

from __future__ import annotations

import heapq
from collections import deque


class _Dinic:
    """Plain max-flow (Dinic) used for feasibility checks."""

    def __init__(self, num_nodes: int) -> None:
        self._graph: list[list[list[int]]] = [[] for _ in range(num_nodes)]

    def add_edge(self, src: int, dst: int, cap: int) -> None:
        self._graph[src].append([dst, cap, len(self._graph[dst])])
        self._graph[dst].append([src, 0, len(self._graph[src]) - 1])

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        num_nodes = len(self._graph)
        while True:
            level = [-1] * num_nodes
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for dst, cap, _ in self._graph[node]:
                    if cap > 0 and level[dst] < 0:
                        level[dst] = level[node] + 1
                        queue.append(dst)
            if level[sink] < 0:
                return flow
            it = [0] * num_nodes

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                edges = self._graph[node]
                while it[node] < len(edges):
                    edge = edges[it[node]]
                    dst, cap, rev = edge
                    if cap > 0 and level[dst] == level[node] + 1:
                        pushed = send(dst, min(amount, cap))
                        if pushed > 0:
                            edge[1] -= pushed
                            self._graph[dst][rev][1] += pushed
                            return pushed
                    it[node] += 1
                return 0

            while True:
                pushed = send(source, 1 << 60)
                if pushed == 0:
                    break
                flow += pushed


class _MinCostFlow:
    """Min-cost max-flow via successive shortest paths (Dijkstra + potentials)."""

    def __init__(self, num_nodes: int) -> None:
        self._graph: list[list[list[int]]] = [[] for _ in range(num_nodes)]

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> None:
        self._graph[src].append([dst, cap, cost, len(self._graph[dst])])
        self._graph[dst].append([src, 0, -cost, len(self._graph[src]) - 1])

    def flow(self, source: int, sink: int, target: int) -> tuple[int, int]:
        """Push up to ``target`` units; returns (flow, cost)."""
        num_nodes = len(self._graph)
        potential = [0] * num_nodes
        total_flow = 0
        total_cost = 0
        while total_flow < target:
            dist = [1 << 60] * num_nodes
            prev_node = [-1] * num_nodes
            prev_edge = [-1] * num_nodes
            dist[source] = 0
            heap = [(0, source)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist[node]:
                    continue
                for idx, (dst, cap, cost, _) in enumerate(self._graph[node]):
                    if cap <= 0:
                        continue
                    nd = d + cost + potential[node] - potential[dst]
                    if nd < dist[dst]:
                        dist[dst] = nd
                        prev_node[dst] = node
                        prev_edge[dst] = idx
                        heapq.heappush(heap, (nd, dst))
            if prev_node[sink] < 0:
                break
            for node in range(num_nodes):
                if dist[node] < (1 << 60):
                    potential[node] += dist[node]
            amount = target - total_flow
            node = sink
            while node != source:
                edge = self._graph[prev_node[node]][prev_edge[node]]
                amount = min(amount, edge[1])
                node = prev_node[node]
            node = sink
            while node != source:
                edge = self._graph[prev_node[node]][prev_edge[node]]
                edge[1] -= amount
                self._graph[node][edge[3]][1] += amount
                total_cost += amount * edge[2]
                node = prev_node[node]
            total_flow += amount
        return total_flow, total_cost

    def used_edges_from(self, src: int) -> list[tuple[int, int]]:
        """Return (dst, units) for edges out of ``src`` carrying flow."""
        result = []
        for dst, cap, _, rev in self._graph[src]:
            used = self._graph[dst][rev][1]
            if used > 0 and cap >= 0:
                result.append((dst, used))
        return result


def _build_left_to_right(
    demands: list[int],
    edges: list[list[int]],
    num_right: int,
) -> tuple[int, int, list[int], list[int]]:
    """Node layout: source, per-unit left nodes, right nodes, sink."""
    num_left = len(demands)
    source = 0
    left_base = 1
    right_base = left_base + num_left
    sink = right_base + num_right
    return source, sink, [left_base + i for i in range(num_left)], [
        right_base + j for j in range(num_right)
    ]


def max_bipartite_matching(
    demands: list[int],
    edges: list[list[int]],
    num_right: int,
) -> bool:
    """Return True if every left node i can receive ``demands[i]`` distinct
    right nodes, each right node used at most once.

    ``edges[i]`` lists the right nodes adjacent to left node ``i``.
    """
    source, sink, left_nodes, right_nodes = _build_left_to_right(
        demands, edges, num_right
    )
    dinic = _Dinic(sink + 1)
    total = 0
    for i, demand in enumerate(demands):
        if demand > 0:
            dinic.add_edge(source, left_nodes[i], demand)
            total += demand
        for j in edges[i]:
            dinic.add_edge(left_nodes[i], right_nodes[j], 1)
    for j in range(num_right):
        dinic.add_edge(right_nodes[j], sink, 1)
    return dinic.max_flow(source, sink) == total


def min_cost_assignment(
    demands: list[int],
    edges: list[list[int]],
    costs: list[dict[int, int]],
    num_right: int,
) -> list[list[int]] | None:
    """Assign each left node i exactly ``demands[i]`` distinct right nodes
    minimizing total cost; each right node used at most once.

    Returns per-left-node lists of chosen right nodes, or None if infeasible.
    """
    source, sink, left_nodes, right_nodes = _build_left_to_right(
        demands, edges, num_right
    )
    mcf = _MinCostFlow(sink + 1)
    total = 0
    for i, demand in enumerate(demands):
        if demand > 0:
            mcf.add_edge(source, left_nodes[i], demand, 0)
            total += demand
        for j in edges[i]:
            mcf.add_edge(left_nodes[i], right_nodes[j], 1, costs[i].get(j, 0))
    for j in range(num_right):
        mcf.add_edge(right_nodes[j], sink, 1, 0)
    flow, _ = mcf.flow(source, sink, total)
    if flow != total:
        return None
    result: list[list[int]] = [[] for _ in demands]
    for i, demand in enumerate(demands):
        if demand == 0:
            continue
        for dst, used in mcf.used_edges_from(left_nodes[i]):
            right_idx = dst - right_nodes[0]
            if 0 <= right_idx < num_right:
                result[i].extend([right_idx] * used)
    return result
