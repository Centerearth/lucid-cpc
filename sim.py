"""Rough local LPC simulator for smoke-testing strategy.py. Approximates the rules; not the real server.

Usage: python3 sim.py [teams]
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import Planner  # noqa: E402

W = H = 50
CENTER = (17, 17, 33, 33)
DIRS = {"north": (0, -1), "north-east": (1, -1), "east": (1, 0), "south-east": (1, 1),
        "south": (0, 1), "south-west": (-1, 1), "west": (-1, 0), "north-west": (-1, -1)}
VISION = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if abs(dx) + abs(dy) <= 3]


def in_center(x, y):
    return CENTER[0] <= x < CENTER[2] and CENTER[1] <= y < CENTER[3]


class Sim:
    def __init__(self, seed, n_teams, max_turns=480):
        self.rng = rng = random.Random(seed)
        self.max_turns = max_turns
        self.turn = 0
        self.terrain = [rng.choices(["empty", "mountain", "crater"], [70, 20, 10])[0] for _ in range(W * H)]
        self.teams = [f"t{k}" for k in range(n_teams)]
        self.bank = {t: {"gold": 0, "iron": 15} for t in self.teams}
        self.recruited = {t: 0 for t in self.teams}
        self.bases = {}   # baseId -> (team, (x, y))
        self.units = {}   # unitId -> dict
        self.deposits = {}  # (x, y) -> [resource, amount]
        self.respawns = []  # (resource, old site) due at the start of the next turn
        for t in self.teams:
            spacing, attempts = 10, 0
            while True:
                x, y = rng.randrange(3, W - 3), rng.randrange(3, H - 3)
                if in_center(x, y) or self.terrain[y * W + x] != "empty":
                    continue
                if all(abs(x - bx) + abs(y - by) >= spacing for _, (bx, by) in self.bases.values()):
                    break
                # Crowded maps (e.g. 16 teams) can't keep the full spacing; relax it gradually.
                attempts += 1
                if attempts % 2000 == 0:
                    spacing -= 1
            self.bases[f"b{len(self.bases)}"] = (t, (x, y))
            self.terrain[y * W + x + 1] = "empty"
            self._add_unit(t, (x + 1, y), 0)
        # Three gold and two iron per team; 35% of the gold (rounded) starts in the center,
        # and each base gets one outside gold deposit within its starting vision.
        total_gold = 3 * n_teams
        center_gold = round(0.35 * total_gold)
        for _ in range(center_gold):
            self._place("gold", "center")
        for _, (bx, by) in list(self.bases.values()):
            self._place("gold", "outside", near=(bx, by))
        for _ in range(total_gold - center_gold - n_teams):
            self._place("gold", "outside")
        for _ in range(2 * n_teams):
            self._place("iron", None)

    def _add_unit(self, team, pos, turn):
        uid = f"u{len(self.units)}"
        # streak: consecutive eligible mine attempts on this tile; arrived: turn it reached this tile;
        # key: stable seeded tie-breaker for mining priority.
        self.units[uid] = {"unitId": uid, "teamId": team, "pos": pos, "inv": {"gold": 0, "iron": 0}, "ready": 0,
                           "streak": 0, "arrived": turn, "key": self.rng.random()}

    def _eligible_cells(self, zone):
        occupied = {u["pos"] for u in self.units.values()} | {p for _, p in self.bases.values()}
        return [(x, y) for y in range(H) for x in range(W)
                if self.terrain[y * W + x] == "empty" and (x, y) not in self.deposits and (x, y) not in occupied
                and (zone is None or in_center(x, y) == (zone == "center"))]

    def _place(self, resource, zone, old=None, near=None):
        """Place one deposit using the server's preference order. Returns False if no cell is free."""
        zones = [zone]
        if zone is not None:
            zones.append("outside" if zone == "center" else "center")
        near_ore = {(x + dx, y + dy) for x, y in self.deposits for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        for z in zones:
            cells = self._eligible_cells(z)
            if near is not None:
                cells = [c for c in cells if abs(c[0] - near[0]) + abs(c[1] - near[1]) <= 3] or cells
            if not cells:
                continue
            # At least two movement steps (Chebyshev) from the old site and from other ore,
            # then relax ore spacing, then old-site spacing; the old site itself is the last resort.
            far_from_old = [c for c in cells if old is None or max(abs(c[0] - old[0]), abs(c[1] - old[1])) >= 2]
            tiers = [[c for c in far_from_old if c not in near_ore], far_from_old,
                     [c for c in cells if c != old], cells]
            site = self.rng.choice(next(tier for tier in tiers if tier))
            if resource == "iron":
                amount = self.rng.randint(2, 4)
            else:
                amount = self.rng.randint(12, 18) if in_center(*site) else self.rng.randint(6, 9)
            self.deposits[site] = [resource, amount]
            return True
        return False

    def _respawn(self, resource, old):
        if resource == "gold":
            return self._place("gold", "center" if self.rng.random() < 0.35 else "outside", old=old)
        return self._place("iron", None, old=old)

    def view(self, team):
        visible = set()
        sources = [u["pos"] for u in self.units.values() if u["teamId"] == team]
        sources += [p for t, p in self.bases.values() if t == team]
        for x, y in sources:
            for dx, dy in VISION:
                if 0 <= x + dx < W and 0 <= y + dy < H:
                    visible.add((x + dx, y + dy))
        cells = []
        for y in range(H):
            for x in range(W):
                if (x, y) not in visible:
                    cells.append({"type": "hidden"})
                elif (x, y) in self.deposits:
                    r, a = self.deposits[(x, y)]
                    cells.append({"type": r, "amount": a})
                else:
                    cells.append({"type": self.terrain[y * W + x]})
        units = []
        for u in self.units.values():
            if u["pos"] in visible:
                unit = {"unitId": u["unitId"], "teamId": u["teamId"], "position": {"x": u["pos"][0], "y": u["pos"][1]},
                        "inventoryCapacity": 8, "readyOnTurn": u["ready"]}
                if u["teamId"] == team:
                    unit["inventory"] = dict(u["inv"])
                units.append(unit)
        bases = [{"baseId": b, "teamId": t, "position": {"x": p[0], "y": p[1]}}
                 for b, (t, p) in self.bases.items() if p in visible]
        return {"turn": self.turn, "map": {"width": W, "height": H, "cells": cells}, "units": units,
                "bases": bases, "scores": {t: self.bank[t]["gold"] for t in self.teams},
                "teamResources": {team: dict(self.bank[team])},
                "highGoldArea": {"width": 16, "height": 16, "min": 12, "max": 18, "spawnPercent": 35},
                "warnings": [], "info": [], "effects": []}

    def step(self, actions_by_team):
        acts = {}
        for team, actions in actions_by_team.items():
            for uid, a in actions.items():
                u = self.units.get(uid)
                assert u is not None and u["teamId"] == team, f"{team} sent action for {uid}"
                assert a["type"] in ("move", "mine", "deposit", "trade", "spawn", "build-base", "pass"), a
                if a["type"] == "move":
                    assert a["direction"] in DIRS, a
                if self.turn >= u["ready"]:
                    acts[uid] = a
        new_turn = self.turn + 1
        # Respawns with no free site wait for a later turn.
        self.respawns = [(r, old) for r, old in self.respawns if not self._respawn(r, old)]
        enemy_base = {p: t for t, p in self.bases.values()}
        for uid, a in acts.items():
            if a["type"] != "move":
                continue
            u = self.units[uid]
            dx, dy = DIRS[a["direction"]]
            x, y = u["pos"][0] + dx, u["pos"][1] + dy
            if not (0 <= x < W and 0 <= y < H) or enemy_base.get((x, y), u["teamId"]) != u["teamId"]:
                continue
            u["pos"] = (x, y)
            # Leaving a tile resets both mining and occupancy priority.
            u["streak"] = 0
            u["arrived"] = new_turn
            if self.terrain[y * W + x] in ("mountain", "crater"):
                u["ready"] = new_turn + 2
        order = self.teams[self.turn % len(self.teams):] + self.teams[:self.turn % len(self.teams)]
        for team in order:
            for uid in sorted(acts):
                u, a = self.units[uid], acts[uid]
                if u["teamId"] != team:
                    continue
                if a["type"] == "spawn":
                    a_, b_ = 5, 6
                    for _ in range(self.recruited[team]):
                        a_, b_ = b_, a_ + b_
                    if self.bank[team]["iron"] >= a_:
                        self.bank[team]["iron"] -= a_
                        self.recruited[team] += 1
                        self._add_unit(team, u["pos"], new_turn)
                        self.units[f"u{len(self.units) - 1}"]["ready"] = new_turn
                elif a["type"] == "build-base":
                    x, y = u["pos"]
                    alone = sum(1 for o in self.units.values() if o["pos"] == (x, y)) == 1
                    ok = (self.terrain[y * W + x] == "empty" and (x, y) not in self.deposits and alone
                          and all(p != (x, y) for _, p in self.bases.values())
                          and all(abs(p[0] - x) + abs(p[1] - y) > 2 for t, p in self.bases.values() if t != team))
                    if ok and self.bank[team]["iron"] >= 6:
                        self.bank[team]["iron"] -= 6
                        self.bases[f"b{len(self.bases)}"] = (team, (x, y))
        # Group eligible miners by tile. Any other action, an idle worker, or an
        # ineligible attempt ends the mining streak.
        by_tile = {}
        for uid, u in self.units.items():
            a = acts.get(uid)
            if a is None or a["type"] != "mine":
                if self.turn >= u["ready"]:
                    u["streak"] = 0
                continue
            d = self.deposits.get(u["pos"])
            claim = {t for t, p in self.bases.values() if abs(p[0] - u["pos"][0]) + abs(p[1] - u["pos"][1]) <= 2}
            if (not d or u["inv"]["gold"] + u["inv"]["iron"] >= 8 or (claim and u["teamId"] not in claim)):
                u["streak"] = 0
                continue
            by_tile.setdefault(u["pos"], []).append(u)
        for pos, miners in by_tile.items():
            d = self.deposits[pos]
            # Longest streak first, then longest on the tile, then the stable seeded key.
            miners.sort(key=lambda u: (-u["streak"], u["arrived"], u["key"]))
            for u in miners:
                if d[1] > 0:
                    d[1] -= 1
                    u["inv"][d[0]] += 1
                # An eligible attempt counts even if another worker took the last ore.
                u["streak"] += 1
            if d[1] == 0:
                del self.deposits[pos]
                self.respawns.append((d[0], pos))
        for uid, a in acts.items():
            if a["type"] == "trade":
                inv, src = self.units[uid]["inv"], a["from"]
                dst = "gold" if src == "iron" else "iron"
                if inv[src] >= 3:
                    inv[src] -= 3
                    inv[dst] += 1
        for uid, a in acts.items():
            u = self.units[uid]
            if a["type"] == "deposit" and enemy_base.get(u["pos"]) == u["teamId"]:
                for r in ("gold", "iron"):
                    self.bank[u["teamId"]][r] += u["inv"][r]
                    u["inv"][r] = 0
        self.turn = new_turn


def run(seed, n_teams):
    sim = Sim(seed, n_teams)
    planners = {t: Planner(t, "sim", W, H) for t in sim.teams}
    worst = 0.0
    while sim.turn < sim.max_turns:
        actions = {}
        for t, p in planners.items():
            started = time.monotonic()
            actions[t] = p.decide(sim.view(t), sim.max_turns)
            worst = max(worst, time.monotonic() - started)
        sim.step(actions)
    carried = {t: sum(u["inv"]["gold"] for u in sim.units.values() if u["teamId"] == t) for t in sim.teams}
    workers = {t: sum(1 for u in sim.units.values() if u["teamId"] == t) for t in sim.teams}
    bases = {t: sum(1 for bt, _ in sim.bases.values() if bt == t) for t in sim.teams}
    print(f"seed {seed}: worst turn {worst * 1000:.0f} ms")
    for t in sim.teams:
        print(f"  {t}: banked gold {sim.bank[t]['gold']:4d}  iron {sim.bank[t]['iron']:3d}  "
              f"carried gold {carried[t]}  workers {workers[t]}  bases {bases[t]}")


if __name__ == "__main__":
    for seed in (1, 2, 3, 4):
        run(seed, int(sys.argv[1]) if len(sys.argv) > 1 else 6)
