"""Layered LPC strategy: world memory, distance fields, and greedy task assignment.

Every tunable number lives in WEIGHTS so an offline tuner can search over them.
"""
from __future__ import annotations

import time
from heapq import nlargest
from typing import Optional

from bot_client import Action, ActionsByEntity, BotState, Identity, Inventory, MarsView, Unit, mars_view
from memory import EMPTY, OBSTACLE, UNKNOWN, VOID, Deposit, WorldMemory
from navigation import INF, OBSTACLE_COST, dijkstra, dijkstra_to, direction, first_step

BASE_COST = 6
TRADE_RATIO = 3
VISION = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if abs(dx) + abs(dy) <= 3]

# Tuned values come from tune.py candidate 3 (16 teams, +22.9 ± 11.7 gold over 12 games).
WEIGHTS: dict[str, float] = {
    # Expected entry cost of an unseen tile: 70% plain, 30% mountain or crater.
    "unknown_cost": 2.09,
    # Value of one kg of iron in gold, while we want to invest and once we don't.
    "iron_value_needed": 1.854,
    "iron_value_spare": 1 / 3,
    # Gold-equivalent cost of one banked kg of iron when deciding to spend it.
    "iron_gold_equiv": 3.0,
    "worker_gold_per_turn": 0.504,
    "spawn_warmup_turns": 15,
    "spawn_on_deposit_min": 4,
    "max_workers": 12,
    "iron_reserve": 0,
    "invest_min_turns_left": 60,
    # Skip non-urgent bases that would delay a paying worker while we have fewer workers than this.
    "save_workers_until": 3,
    # Spawn wherever the worker stands before this turn (2 = spawn on turns 0 and 1).
    "spawn_anywhere_until_turn": 2,
    # 1 = trade carried gold for iron at a base when a paying worker is short of iron.
    "trade_gold_for_workers": 0,
    # Build a base instead of walking home when it saves at least this many turns.
    "base_min_savings": 10,
    "base_min_turns_left": 40,
    # 1 = build a base next to a far, rich gold deposit before mining it, not after the first load.
    "base_first": 0,
    "base_first_min_amount": 8,
    # 1 = send a worker to build next to known gold clusters far from our bases.
    "base_cluster": 1,
    "base_cluster_min_gold": 15,
    "base_cluster_min_dist": 5,
    "build_fail_limit": 2,
    "explore_value": 3.775,
    "explore_dist_offset": 6.0,
    "explore_candidates": 15,
    "explore_spread": 7,
    "explore_spread_penalty": 0.3,
    "explore_min_turns_left": 37,
    "center_bias": 0.5,
    # Extra explore distance per tile of Chebyshev distance from the center point.
    "explore_center_pull": 0.305,
    # 1 = explore only when a worker has no mining job at all.
    "explore_last": 1,
    # 1 = carried gold counts in job values, and going home competes as a job.
    "value_carried": 1,
    # 1 = send the worker closest to known iron whenever the bank is short of the next build.
    "iron_runner": 1,
    "stale_turns": 194.7,
    "contention_penalty": 0.79,
    "stickiness": 1.535,
    # Carrying workers go home when their best task scores below this.
    "return_threshold": 0.15,
    "endgame_buffer": 2,
    "iron_dump_turns_left": 60,
    "blacklist_turns": 60,
}

Task = tuple[str, int]


def spawn_cost(recruited: int) -> int:
    a, b = 5, 6
    for _ in range(recruited):
        a, b = b, a + b
    return a


def inventory(unit: Unit) -> Inventory:
    return unit.get("inventory") or {"gold": 0, "iron": 0}


def carried_kg(unit: Unit) -> int:
    inv = inventory(unit)
    return inv["gold"] + inv["iron"]


class Planner:
    """State that persists across the turns of one match."""

    def __init__(self, team_id: str, match_id: str, width: int, height: int,
                 weights: Optional[dict[str, float]] = None) -> None:
        self.match_id = match_id
        self.memory = WorldMemory(team_id, width, height)
        self.weights = {**WEIGHTS, **(weights or {})}
        self.assignments: dict[str, Task] = {}
        self.build_intents: dict[str, int] = {}
        self.last_mines: dict[str, tuple[int, int]] = {}
        self.last_builds: list[int] = []
        self.build_failures: dict[int, int] = {}
        self.bad_sites: set[int] = set()
        self.iron_runner: Optional[str] = None

    def decide(self, view: MarsView, max_turns: int) -> ActionsByEntity:
        started = time.monotonic()
        self.memory.update(view)
        self._detect_failed_mines()
        self._detect_failed_builds()
        plan = TurnPlan(self, view, max_turns)
        actions = plan.run()
        self.assignments = plan.assignments
        self.last_mines = plan.mines
        self.last_builds = plan.builds
        elapsed = time.monotonic() - started
        if elapsed > 0.4:
            print(f"Planning turn {view['turn']} took {elapsed * 1000:.0f} ms")
        return actions

    def _detect_failed_mines(self) -> None:
        mem = self.memory
        for worker in mem.my_units:
            last = self.last_mines.get(worker["unitId"])
            if last is None or mem.position(worker) != last[0]:
                continue
            cell, carried_before = last
            # Ore remained and the worker had room, so the attempt was ineligible,
            # most likely because a hidden enemy base claims the deposit.
            if (cell in mem.deposits and mem.visible[cell]
                    and carried_kg(worker) <= carried_before < worker["inventoryCapacity"]):
                mem.blacklist[cell] = mem.turn + int(self.weights["blacklist_turns"])

    def _detect_failed_builds(self) -> None:
        # A build that left no base fails for a reason we can't see; stop retrying that site.
        for site in self.last_builds:
            if site not in self.memory.friendly_bases:
                self.build_failures[site] = self.build_failures.get(site, 0) + 1
                if self.build_failures[site] >= self.weights["build_fail_limit"]:
                    self.bad_sites.add(site)


class TurnPlan:
    """One turn's decisions. Distance fields are rebuilt each turn from memory."""

    def __init__(self, planner: Planner, view: MarsView, max_turns: int) -> None:
        self.planner = planner
        self.mem = mem = planner.memory
        self.w = planner.weights
        self.turns_left = max_turns - view["turn"]
        bank = view["teamResources"].get(mem.team_id, {"gold": 0, "iron": 0})
        self.iron_budget = bank["iron"]
        self.workers = sorted(mem.my_units, key=lambda unit: unit["unitId"])
        self.spawn_cost = spawn_cost(len(self.workers) - 1)
        # Iron promised to workers walking to a planned base site.
        self.reserved_iron = BASE_COST * len(planner.build_intents)
        investing = self.turns_left > self.w["invest_min_turns_left"]
        self.iron_target = BASE_COST + (self.spawn_cost if self._spawn_pays() else 0)
        self.iron_needed = investing and self.iron_budget < self.iron_target
        self.iron_value = self.w["iron_value_needed"] if self.iron_needed else self.w["iron_value_spare"]

        self.cost = self._enter_costs()
        self.base_dist = dijkstra_to(mem.grid, self.cost, mem.friendly_bases)
        self.enemy_xy = [mem.grid.xy(mem.position(unit)) for unit in mem.enemy_units]
        self.enemy_base_xy = [mem.grid.xy(i) for i in mem.enemy_bases]
        self.paths: dict[str, tuple[list[float], list[int]]] = {}

        self.actions: ActionsByEntity = {}
        self.assignments: dict[str, Task] = {}
        self.mines: dict[str, tuple[int, int]] = {}
        self.builds: list[int] = []
        self.spawned = False
        # Iron from gold trades submitted this turn; it reaches the bank after the next deposit.
        self.pending_iron = 0

    def run(self) -> ActionsByEntity:
        pending: list[Unit] = []
        for worker in self.workers:
            if self.mem.turn < worker["readyOnTurn"]:
                continue
            # A bug in one worker's logic only costs that worker its turn.
            try:
                action = self._local_action(worker)
            except Exception as error:
                print(f"Turn {self.mem.turn}: worker {worker['unitId']} failed: {error!r}; passing")
                action = {"type": "pass"}
            if action is None:
                pending.append(worker)
            else:
                self.actions[worker["unitId"]] = action
        try:
            self._assign(pending)
        except Exception as error:
            print(f"Turn {self.mem.turn}: assignment failed: {error!r}; unassigned workers pass")
        for worker in pending:
            self.actions.setdefault(worker["unitId"], {"type": "pass"})
        return self.actions

    # Distance fields

    def _enter_costs(self) -> list[float]:
        by_terrain = {UNKNOWN: self.w["unknown_cost"], EMPTY: 1.0, OBSTACLE: float(OBSTACLE_COST), VOID: INF}
        cost = [by_terrain[t] for t in self.mem.terrain]
        for i in self.mem.enemy_bases:
            cost[i] = INF
        return cost

    def path(self, worker: Unit) -> tuple[list[float], list[int]]:
        uid = worker["unitId"]
        if uid not in self.paths:
            self.paths[uid] = dijkstra(self.mem.grid, self.cost, [self.mem.position(worker)])
        return self.paths[uid]

    # Rule-based decisions that don't need assignment

    def _local_action(self, worker: Unit) -> Optional[Action]:
        w, mem = self.w, self.mem
        uid = worker["unitId"]
        pos = mem.position(worker)
        inv = inventory(worker)
        carried = inv["gold"] + inv["iron"]
        space = worker["inventoryCapacity"] - carried
        # Actions to spare before the last trip home has to start.
        slack = self.turns_left - self.base_dist[pos] - 1

        site = self.planner.build_intents.pop(uid, None)
        if site is not None:
            self.reserved_iron -= BASE_COST
            if site == pos and self._free_iron() >= BASE_COST and self._valid_site(pos, builder_here=True):
                return self._build_base(pos)
            if site != pos and self._valid_site(site):
                return self._plan_build(worker, site)

        if mem.turn < w["spawn_anywhere_until_turn"] and self._can_spawn():
            return self._spawn()

        if pos in mem.friendly_bases:
            if carried:
                if inv["iron"] >= TRADE_RATIO and self._dump_iron() and self.turns_left >= 2:
                    return {"type": "trade", "from": "iron"}
                if inv["gold"] >= TRADE_RATIO and self._iron_short(inv):
                    self.pending_iron += 1
                    return {"type": "trade", "from": "gold"}
                return {"type": "deposit"}
            if self._can_spawn():
                return self._spawn()

        if carried and slack <= w["endgame_buffer"]:
            return self._return_home(worker, pos)
        if space <= 0:
            if inv["iron"] >= TRADE_RATIO and not self.iron_needed:
                # Trading frees 2 kg of space, so the worker can keep mining.
                return {"type": "trade", "from": "iron"}
            return self._return_home(worker, pos)

        deposit = mem.deposits.get(pos)
        if (deposit and mem.mineable(pos) and slack > w["endgame_buffer"]
                and (deposit.resource == "gold" or self.iron_needed
                     or self.planner.assignments.get(uid) == ("deposit", pos))):
            if deposit.resource == "gold" and deposit.amount >= w["spawn_on_deposit_min"] and self._can_spawn():
                return self._spawn()
            return self._mine(worker, pos)
        return None

    def _return_home(self, worker: Unit, pos: int) -> Action:
        """Walk to the nearest base, or build one here if that is much faster."""
        w = self.w
        walk = self.base_dist[pos] + 1
        if self._free_iron() >= BASE_COST:
            options = [(2.0, pos)] if self._valid_site(pos, builder_here=True) else []
            options += [(self.cost[n] + 2, n) for n in self.mem.grid.neighbors[pos] if self._valid_site(n)]
            if options:
                alt, site = min(options)
                worth_it = (walk - alt >= w["base_min_savings"] and self.turns_left >= w["base_min_turns_left"]
                            and self._base_affordable())
                # Late in the match, a base may be the only way to bank this load.
                rescue = walk + w["endgame_buffer"] > self.turns_left >= alt
                if worth_it or rescue:
                    return self._build_base(pos) if site == pos else self._plan_build(worker, site)
        return self._step_home(pos)

    def _step_home(self, pos: int) -> Action:
        neighbors = self.mem.grid.neighbors[pos]
        best = min(neighbors, key=lambda n: self.cost[n] + self.base_dist[n])
        if self.cost[best] + self.base_dist[best] == INF:
            return {"type": "pass"}
        return {"type": "move", "direction": direction(self.mem.grid, pos, best)}

    def _move_towards(self, worker: Unit, target: int) -> Action:
        pos = self.mem.position(worker)
        step = first_step(self.path(worker)[1], pos, target)
        if step is None:
            return {"type": "pass"}
        return {"type": "move", "direction": direction(self.mem.grid, pos, step)}

    def _mine(self, worker: Unit, pos: int) -> Action:
        self.mines[worker["unitId"]] = (pos, carried_kg(worker))
        self.assignments[worker["unitId"]] = ("deposit", pos)
        return {"type": "mine"}

    # Investment

    def _spawn_pays(self) -> bool:
        w = self.w
        payoff = w["worker_gold_per_turn"] * (self.turns_left - w["spawn_warmup_turns"])
        return (len(self.workers) < w["max_workers"]
                and self.turns_left > w["invest_min_turns_left"]
                and payoff > self.spawn_cost * w["iron_gold_equiv"])

    def _free_iron(self) -> int:
        return self.iron_budget - self.reserved_iron

    def _can_spawn(self) -> bool:
        return (not self.spawned and self._spawn_pays()
                and self._free_iron() - self.spawn_cost >= self.w["iron_reserve"])

    def _spawn(self) -> Action:
        self.spawned = True
        self.iron_budget -= self.spawn_cost
        return {"type": "spawn"}

    def _base_affordable(self) -> bool:
        """Bank iron covers a base, plus the next worker when saving for one."""
        saving = len(self.workers) < self.w["save_workers_until"] and self._spawn_pays()
        return self._free_iron() >= BASE_COST + (self.spawn_cost if saving else 0)

    def _iron_short(self, inv: Inventory) -> bool:
        return (bool(self.w["trade_gold_for_workers"]) and self._spawn_pays()
                and self.iron_budget + self.pending_iron + inv["iron"] < self.spawn_cost)

    def _build_base(self, site: int) -> Action:
        self.iron_budget -= BASE_COST
        self.builds.append(site)
        return {"type": "build-base"}

    def _plan_build(self, worker: Unit, site: int) -> Action:
        """Head for a base site, holding its iron until the worker gets there."""
        self.planner.build_intents[worker["unitId"]] = site
        self.reserved_iron += BASE_COST
        return self._move_towards(worker, site)

    def _dump_iron(self) -> bool:
        return not self.iron_needed and self.turns_left <= self.w["iron_dump_turns_left"]

    def _valid_site(self, i: int, builder_here: bool = False) -> bool:
        mem = self.mem
        if mem.terrain[i] != EMPTY or i in mem.deposits or i in mem.base_cells or i in self.planner.bad_sites:
            return False
        if mem.unit_counts[i] > (1 if builder_here else 0):
            return False
        x, y = mem.grid.xy(i)
        return all(abs(bx - x) + abs(by - y) > 2 for bx, by in self.enemy_base_xy)

    # Task assignment

    def _assign(self, workers: list[Unit]) -> None:
        """Greedy assignment: repeatedly give the best-scoring (worker, task) pair its task."""
        reserved: dict[int, int] = {}
        explore_targets: list[int] = []
        remaining = list(workers)
        for special in (self._send_cluster_builder, self._send_iron_runner):
            worker = special(remaining)
            if worker is not None:
                remaining.remove(worker)
        frontier = self._frontier() if self.turns_left > self.w["explore_min_turns_left"] else []
        options = {worker["unitId"]: self._explore_options(worker, frontier) for worker in remaining}
        while remaining:
            best: Optional[tuple[float, Unit, Task, int]] = None
            for worker in remaining:
                option = self._best_task(worker, reserved, explore_targets, options[worker["unitId"]])
                if option is not None and (best is None or option[0] > best[0]):
                    best = (option[0], worker, option[1], option[2])
            if best is None:
                break
            score, worker, task, gain = best
            remaining.remove(worker)
            uid = worker["unitId"]
            pos = self.mem.position(worker)
            kind, target = task
            if kind == "return" or (not self.w["value_carried"] and carried_kg(worker)
                                    and score < self.w["return_threshold"]):
                self.actions[uid] = self._return_home(worker, pos)
                continue
            self.assignments[uid] = task
            if kind == "deposit":
                reserved[target] = reserved.get(target, 0) + gain
                site = None if target == pos else self._base_first_site(worker, target)
                if site == pos:
                    self.actions[uid] = self._build_base(pos)
                elif site is not None:
                    self.actions[uid] = self._plan_build(worker, site)
                else:
                    self.actions[uid] = self._mine(worker, pos) if target == pos else self._move_towards(worker, target)
            else:
                explore_targets.append(target)
                self.actions[uid] = self._move_towards(worker, target)
        for worker in remaining:
            pos = self.mem.position(worker)
            self.actions[worker["unitId"]] = self._return_home(worker, pos) if carried_kg(worker) else {"type": "pass"}

    def _base_first_site(self, worker: Unit, i: int) -> Optional[int]:
        """A site next to deposit i to build on before mining it, or None to just go mine."""
        w, mem = self.w, self.mem
        deposit = mem.deposits[i]
        if (not w["base_first"] or deposit.resource != "gold" or deposit.amount < w["base_first_min_amount"]
                or self.base_dist[i] + 1 - 3 < w["base_min_savings"]
                or self.turns_left < w["base_min_turns_left"] or not self._base_affordable()):
            return None
        # Another worker is already on its way to build near this deposit.
        if any(mem.grid.manhattan(i, s) <= 2 for s in self.planner.build_intents.values()):
            return None
        pos = mem.position(worker)
        dist = self.path(worker)[0]
        sites = [n for n in mem.grid.neighbors[i] if dist[n] < INF and self._valid_site(n, builder_here=n == pos)]
        return min(sites, key=lambda n: dist[n]) if sites else None

    def _send_cluster_builder(self, workers: list[Unit]) -> Optional[Unit]:
        """Send one worker to build next to the best known gold cluster far from our bases."""
        w, mem = self.w, self.mem
        if (not w["base_cluster"] or self.planner.build_intents or not workers
                or self.turns_left < w["base_min_turns_left"] or not self._base_affordable()):
            return None
        golds = [(i, d.amount) for i, d in mem.deposits.items() if d.resource == "gold" and mem.mineable(i)]
        choices: list[tuple[float, int, Unit]] = []
        for i, _ in golds:
            if self.base_dist[i] < w["base_cluster_min_dist"]:
                continue
            nearby = sum(amount for j, amount in golds if mem.grid.chebyshev(i, j) <= 3)
            if nearby < w["base_cluster_min_gold"]:
                continue
            for worker in workers:
                travel = self.path(worker)[0][i]
                if travel < self.turns_left - 50:
                    choices.append((nearby / (travel + 5), i, worker))
        for _, i, worker in sorted(choices, key=lambda choice: -choice[0]):
            site = self._best_site(i, worker, golds)
            if site is not None:
                pos = mem.position(worker)
                self.actions[worker["unitId"]] = self._build_base(pos) if site == pos else self._plan_build(worker, site)
                return worker
        return None

    def _best_site(self, center: int, worker: Unit, golds: list[tuple[int, int]]) -> Optional[int]:
        """Site within 3 of center scoring claimed gold (Manhattan 2) plus half the gold within 3 steps."""
        w, mem = self.w, self.mem
        grid = mem.grid
        pos = mem.position(worker)
        dist = self.path(worker)[0]
        cx, cy = grid.xy(center)
        best, best_score = None, -INF
        for y in range(max(0, cy - 3), min(grid.height, cy + 4)):
            for x in range(max(0, cx - 3), min(grid.width, cx + 4)):
                site = grid.index(x, y)
                if (dist[site] == INF or self.base_dist[site] < w["base_cluster_min_dist"]
                        or not self._valid_site(site, builder_here=site == pos)):
                    continue
                score = -0.5 * grid.chebyshev(pos, site)
                for j, amount in golds:
                    if grid.manhattan(site, j) <= 2:
                        score += amount
                    elif grid.chebyshev(site, j) <= 3:
                        score += 0.5 * amount
                if score > best_score:
                    best, best_score = site, score
        return best

    def _send_iron_runner(self, workers: list[Unit]) -> Optional[Unit]:
        """While the bank is short of the next build, the worker closest to known iron fetches it."""
        mem = self.mem
        if not self.w["iron_runner"] or not self.iron_needed or len(self.workers) < 2:
            self.planner.iron_runner = None
            return None
        runner = self.planner.iron_runner
        free_ids = {worker["unitId"] for worker in workers}
        if runner in {worker["unitId"] for worker in self.workers} and runner not in free_ids:
            return None  # The runner is busy with a fixed rule this turn (mining, returning, ...).
        irons = [i for i, d in mem.deposits.items() if d.resource == "iron" and mem.mineable(i)]
        pool = [worker for worker in workers if worker["unitId"] == runner] or workers
        options = [(self.path(worker)[0][i], i, worker) for worker in pool for i in irons]
        if not options:
            return None
        travel, i, worker = min(options, key=lambda option: option[0])
        if travel >= self.turns_left - 30:
            return None
        uid = worker["unitId"]
        pos = mem.position(worker)
        self.planner.iron_runner = uid
        if self.iron_budget + inventory(worker)["iron"] >= self.iron_target:
            self.actions[uid] = self._return_home(worker, pos)
        else:
            self.assignments[uid] = ("deposit", i)
            self.actions[uid] = self._mine(worker, pos) if i == pos else self._move_towards(worker, i)
        return worker

    def _best_task(self, worker: Unit, reserved: dict[int, int], explore_targets: list[int],
                   explore_options: list[tuple[float, int]]) -> Optional[tuple[float, Task, int]]:
        w, mem = self.w, self.mem
        dist = self.path(worker)[0]
        previous = self.planner.assignments.get(worker["unitId"])
        space = worker["inventoryCapacity"] - carried_kg(worker)
        cheap_base = self._base_affordable() and self.turns_left >= w["base_min_turns_left"]
        best: Optional[tuple[float, Task, int]] = None
        carried_value = 0.0
        if w["value_carried"]:
            inv = inventory(worker)
            carried_value = inv["gold"] + inv["iron"] * self.iron_value
            pos = mem.position(worker)
            if carried_value and self.base_dist[pos] < INF:
                best = (carried_value / (self.base_dist[pos] + 1), ("return", pos), 0)

        for i, deposit in mem.deposits.items():
            if dist[i] == INF or not mem.mineable(i):
                continue
            walk = self.base_dist[i] + 1
            # A far deposit gets a base next to it, so the trip home is about 3 turns.
            back = 3.0 if cheap_base and walk - 3 >= w["base_min_savings"] else walk
            spare = self.turns_left - dist[i] - back - w["endgame_buffer"]
            if spare < 1:
                continue
            gain = min(deposit.amount - reserved.get(i, 0), space, int(spare))
            if gain <= 0:
                continue
            value = carried_value + gain * (1.0 if deposit.resource == "gold" else self.iron_value)
            score = value / (dist[i] + gain + back) * self._freshness(i, deposit) * self._contention(i)
            if previous == ("deposit", i):
                score *= w["stickiness"]
            if best is None or score > best[0]:
                best = (score, ("deposit", i), gain)

        if w["explore_last"] and best is not None:
            return best
        for score, i in explore_options:
            if any(mem.grid.manhattan(i, t) <= w["explore_spread"] for t in explore_targets):
                score *= w["explore_spread_penalty"]
            if previous == ("explore", i):
                score *= w["stickiness"]
            if best is None or score > best[0]:
                best = (score, ("explore", i), 0)
        return best

    def _freshness(self, i: int, deposit: Deposit) -> float:
        if self.mem.visible[i]:
            return 1.0
        return 1.0 / (1.0 + (self.mem.turn - deposit.seen_turn) / self.w["stale_turns"])

    def _contention(self, i: int) -> float:
        x, y = self.mem.grid.xy(i)
        rivals = sum(1 for ex, ey in self.enemy_xy if max(abs(ex - x), abs(ey - y)) <= 2)
        return 1.0 / (1.0 + self.w["contention_penalty"] * rivals)

    def _frontier(self) -> list[tuple[int, float]]:
        """Unknown cells with the fraction of their vision diamond that is still unknown."""
        mem = self.mem
        center_bonus = 1.0 + self.w["center_bias"]
        unknown_near, center = mem.unknown_near, mem.center_cells
        return [(i, unknown_near[i] / len(VISION) * (center_bonus if center[i] else 1.0))
                for i, t in enumerate(mem.terrain) if t == UNKNOWN]

    def _explore_options(self, worker: Unit, frontier: list[tuple[int, float]]) -> list[tuple[float, int]]:
        w, mem = self.w, self.mem
        dist = self.path(worker)[0]
        pull = w["explore_center_pull"]
        center = None
        if pull and mem.center:
            x0, y0, x1, y1 = mem.center
            center = mem.grid.index((x0 + x1) // 2, (y0 + y1) // 2)
        scored = ((w["explore_value"] * info / (dist[i] + w["explore_dist_offset"]
                                                + (pull * mem.grid.chebyshev(i, center) if center is not None else 0)), i)
                  for i, info in frontier if dist[i] < INF)
        return nlargest(int(w["explore_candidates"]), scored)


_planner: Optional[Planner] = None


def calculate_actions(gameState: BotState, identity: Identity) -> ActionsByEntity:
    global _planner
    view = mars_view(gameState)
    if view is None:
        return {}
    match_id = gameState["match"]["matchId"]
    if _planner is None or _planner.match_id != match_id:
        game_map = view["map"]
        _planner = Planner(identity["teamId"], match_id, game_map["width"], game_map["height"])
    return _planner.decide(view, gameState["match"]["maxTurns"])
