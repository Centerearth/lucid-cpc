from __future__ import annotations

from heapq import heapify, heappop, heappush
from typing import Optional

# Matches the default obstacleTraversalCost: the entry turn plus two cooldown turns.
OBSTACLE_COST = 3
INF = float("inf")

DIRECTIONS = {
    (0, -1): "north",
    (1, -1): "north-east",
    (1, 0): "east",
    (1, 1): "south-east",
    (0, 1): "south",
    (-1, 1): "south-west",
    (-1, 0): "west",
    (-1, -1): "north-west",
}


class Grid:
    """Flat cell indices (y * width + x) with a precomputed 8-neighbor table."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.size = width * height
        self.neighbors: list[list[int]] = [
            [
                (i // width + dy) * width + i % width + dx
                for dx, dy in DIRECTIONS
                if 0 <= i % width + dx < width and 0 <= i // width + dy < height
            ]
            for i in range(self.size)
        ]

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def xy(self, i: int) -> tuple[int, int]:
        return i % self.width, i // self.width

    def manhattan(self, a: int, b: int) -> int:
        return abs(a % self.width - b % self.width) + abs(a // self.width - b // self.width)


def dijkstra(grid: Grid, enter_cost: list[float], sources: list[int]) -> tuple[list[float], list[int]]:
    """Turns until ready on each cell, starting from any source, plus predecessors."""
    dist = [INF] * grid.size
    prev = [-1] * grid.size
    heap = [(0.0, s) for s in sources]
    for s in sources:
        dist[s] = 0.0
    heapify(heap)
    neighbors = grid.neighbors
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        for v in neighbors[u]:
            nd = d + enter_cost[v]
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heappush(heap, (nd, v))
    return dist, prev


def dijkstra_to(grid: Grid, enter_cost: list[float], targets: set[int]) -> list[float]:
    """Turns needed to reach the nearest target from each cell (reverse search)."""
    dist = [INF] * grid.size
    heap = [(0.0, t) for t in targets]
    for t in targets:
        dist[t] = 0.0
    heapify(heap)
    neighbors = grid.neighbors
    while heap:
        d, v = heappop(heap)
        if d > dist[v]:
            continue
        # Stepping from u onto v costs v's entry cost.
        nd = d + enter_cost[v]
        for u in neighbors[v]:
            if nd < dist[u]:
                dist[u] = nd
                heappush(heap, (nd, u))
    return dist


def first_step(prev: list[int], start: int, target: int) -> Optional[int]:
    """The cell to move onto next on the way to target, or None if unreachable or already there."""
    if target == start or prev[target] == -1:
        return None
    node = target
    while prev[node] != start:
        node = prev[node]
    return node


def direction(grid: Grid, start: int, step: int) -> str:
    (sx, sy), (tx, ty) = grid.xy(start), grid.xy(step)
    return DIRECTIONS[(tx - sx, ty - sy)]
