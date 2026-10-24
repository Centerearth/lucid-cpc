from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from bot_client import MarsView, Unit
from navigation import Grid

UNKNOWN, EMPTY, OBSTACLE, VOID = 0, 1, 2, 3
TERRAIN = {"empty": EMPTY, "mountain": OBSTACLE, "crater": OBSTACLE, "void": VOID}


@dataclass
class Deposit:
    resource: str
    amount: int
    seen_turn: int


class WorldMemory:
    """Everything seen so far. The live view only covers current vision."""

    def __init__(self, team_id: str, width: int, height: int) -> None:
        self.team_id = team_id
        self.grid = Grid(width, height)
        self.turn = -1
        self.terrain = [UNKNOWN] * self.grid.size
        self.seen_turn = [-1] * self.grid.size
        self.visible = [False] * self.grid.size
        self.deposits: dict[int, Deposit] = {}
        # Bases are never destroyed, so every base ever seen stays valid.
        self.bases: dict[str, tuple[str, int]] = {}
        self.friendly_bases: set[int] = set()
        self.enemy_bases: set[int] = set()
        self.base_cells: set[int] = set()
        self.my_units: list[Unit] = []
        self.enemy_units: list[Unit] = []
        self.unit_counts: Counter[int] = Counter()
        self.enemy_sightings: dict[str, tuple[int, int]] = {}
        # Deposits whose mine attempts failed while ore remained, mapped to the turn they expire.
        self.blacklist: dict[int, int] = {}
        self.center: Optional[tuple[int, int, int, int]] = None
        self.center_cells = [False] * self.grid.size
        # Cells within vision range (Manhattan 3) of each cell, and how many of them are still unknown.
        self.vision_cells = [
            [ny * width + nx
             for dx in range(-3, 4) for dy in range(-3, 4)
             if abs(dx) + abs(dy) <= 3
             for nx, ny in [(i % width + dx, i // width + dy)]
             if 0 <= nx < width and 0 <= ny < height]
            for i in range(self.grid.size)
        ]
        self.unknown_near = [len(cells) for cells in self.vision_cells]

    def update(self, view: MarsView) -> None:
        self.turn = view["turn"]
        visible = [False] * self.grid.size
        for i, cell in enumerate(view["map"]["cells"]):
            kind = cell["type"]
            if kind == "hidden":
                continue
            visible[i] = True
            self.seen_turn[i] = self.turn
            if self.terrain[i] == UNKNOWN and (kind in ("gold", "iron") or kind in TERRAIN):
                for j in self.vision_cells[i]:
                    self.unknown_near[j] -= 1
            if kind in ("gold", "iron"):
                self.terrain[i] = EMPTY
                amount = cell.get("amount", 0)
                if amount > 0:
                    self.deposits[i] = Deposit(kind, amount, self.turn)
                else:
                    self.deposits.pop(i, None)
            elif kind in TERRAIN:
                self.terrain[i] = TERRAIN[kind]
                self.deposits.pop(i, None)
        self.visible = visible

        for base in view["bases"]:
            position = base["position"]
            self.bases[base["baseId"]] = (base["teamId"], self.grid.index(position["x"], position["y"]))
        self.friendly_bases = {i for team, i in self.bases.values() if team == self.team_id}
        self.enemy_bases = {i for team, i in self.bases.values() if team != self.team_id}
        self.base_cells = self.friendly_bases | self.enemy_bases

        self.my_units = [unit for unit in view["units"] if unit["teamId"] == self.team_id]
        self.enemy_units = [unit for unit in view["units"] if unit["teamId"] != self.team_id]
        self.unit_counts = Counter(self.position(unit) for unit in view["units"])
        for unit in self.enemy_units:
            self.enemy_sightings[unit["unitId"]] = (self.position(unit), self.turn)

        area = view.get("highGoldArea")
        if area and self.center is None:
            x0 = (self.grid.width - area["width"]) // 2
            y0 = (self.grid.height - area["height"]) // 2
            self.center = (x0, y0, x0 + area["width"], y0 + area["height"])
            self.center_cells = [self.in_center(i) for i in range(self.grid.size)]

    def position(self, unit: Unit) -> int:
        return self.grid.index(unit["position"]["x"], unit["position"]["y"])

    def in_center(self, i: int) -> bool:
        if self.center is None:
            return False
        x, y = self.grid.xy(i)
        x0, y0, x1, y1 = self.center
        return x0 <= x < x1 and y0 <= y < y1

    def claimers(self, i: int) -> set[str]:
        """Teams with a known base within Manhattan distance 2. Hidden bases can also claim."""
        return {team for team, base in self.bases.values() if self.grid.manhattan(i, base) <= 2}

    def mineable(self, i: int) -> bool:
        if self.blacklist.get(i, -1) >= self.turn:
            return False
        teams = self.claimers(i)
        return not teams or self.team_id in teams
