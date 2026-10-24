#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from heapq import heapify, heappop, heappush
from collections.abc import Sequence
from typing import Optional
from bot_client import (
    ActionsByEntity,
    BaseLocation,
    BotClient,
    BotState,
    Coordinate,
    CoordinateGrid,
    GameMap,
    Identity,
    MarsView,
    MovementState,
    OreState,
    Resource,
    TerritoryState,
    mars_view,
)

# Make sure to set your team token!
API_KEY = os.environ.get("API_KEY", "replace-with-api-key")
API_BASE_URL = os.environ.get("LUCID_API_BASE_URL", "http://localhost:5173")


# Strategy settings. See marissa-spec.md for what each one does.
OBSTACLE_COST = 3          # turns to enter a mountain or crater
HIDDEN_COST = 1.6          # estimated cost of an unseen tile (30% obstacle chance)
BASE_COST = 6
MAX_WORKERS = 5
STOP_SPAWN_TURN = 300
STOP_BASE_TURN = 400
ENDGAME_MARGIN = 3         # extra turns of safety when heading home at the end
DETOUR_LIMIT = 2           # center walkers mine gold this many steps off the path
HYSTERESIS = 1.2           # only switch targets if the new one is 20% better
STALE_TURNS = 50           # remembered deposits older than this count for less
STALE_FACTOR = 0.7
CENTER_NEAR_HOME = 8       # if the center is this close to home, skip the march
CENTER_GIVE_UP_TURN = 80   # stop waiting for the center base after this turn
CENTER_WAIT_LIMIT = 12     # ...or after this many worker-turns spent idle waiting for iron
CENTER_RESERVE_WORKERS = 3 # keep 6 iron for the center base once we have this many workers (2 = spawn only once)
IRON_ROLE_GIVE_UP_TURN = 60  # an iron worker with nothing found goes back to gold after this many turns
NEW_BASE_MIN_DIST = 5      # only build another base if the nearest base is this far
NEW_BASE_MIN_GOLD = 12     # ...and at least this much known gold is nearby
WORKERS_BEFORE_EXTRA_BASES = 4  # extra bases (after the center one) wait until we have this many workers
IRON_WEIGHT = 2.0         # how much 1 kg of iron counts vs gold when we still want to build
EXPLORE_HOME_RADIUS = 8    # the iron worker only explores this close to home
BLOCKED_TURNS = 20         # skip a deposit this long after a mine attempt did nothing
LOG = True

STEPS = {
    (0, -1): "north", (1, -1): "north-east", (1, 0): "east", (1, 1): "south-east",
    (0, 1): "south", (-1, 1): "south-west", (-1, 0): "west", (-1, -1): "north-west",
}
INF = float("inf")
PASS = {"type": "pass"}

# Memory that persists between turns (the client runs in one process).
MEM: dict = {}


def calculate_actions(gameState: BotState, identity: Identity) -> ActionsByEntity:
    view = mars_view(gameState)
    if view is None:
        return {}

    started = time.perf_counter()
    team = identity["teamId"]
    match = gameState["match"]
    if MEM.get("match_id") != match["matchId"]:
        _reset_memory(match["matchId"])

    turn = view["turn"]
    _update_memory(view, turn)
    c = _build_context(view, team, turn, match.get("maxTurns", 480))

    actions: ActionsByEntity = {}
    try:
        _plan(c, actions)
    except Exception as error:
        print(f"T{turn:03d} planning error: {error!r}")

    for worker in c.workers:
        actions.setdefault(worker["unitId"], PASS)
    _remember_actions(c, actions)

    if LOG:
        _log_turn(c, actions, (time.perf_counter() - started) * 1000)
    return actions


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _reset_memory(match_id: str) -> None:
    MEM.clear()
    MEM.update({
        "match_id": match_id,
        "terrain": {},      # (x, y) -> "empty" | "mountain" | "crater" | "void"
        "deposits": {},     # (x, y) -> {"type", "amount", "seen"}
        "bases": {},        # baseId -> {"team", "pos"}; bases never move, so enemy ones stay known
        "workers": {},      # unitId -> role state
        "order": [],        # our unitIds in the order we first saw them
        "home": None,
        "center_base": False,
        "center_given_up": False,
        "center_wait": 0,   # worker-turns spent idle waiting for center-base iron
        "pending": None,    # base build sent last turn: {"uid", "site"}
        "bad_sites": set(),
        "build_tries": {},
        "blocked": {},      # deposit -> turn it becomes usable again
    })


def _update_memory(view: MarsView, turn: int) -> None:
    terrain, deposits = MEM["terrain"], MEM["deposits"]
    width = view["map"]["width"]
    for i, cell in enumerate(view["map"]["cells"]):
        kind = cell.get("type")
        if kind in ("hidden", "unit", None):
            continue
        pos = (i % width, i // width)
        if kind in ("gold", "iron"):
            terrain[pos] = "empty"
            if cell.get("amount", 0) > 0:
                deposits[pos] = {"type": kind, "amount": cell["amount"], "seen": turn}
            else:
                deposits.pop(pos, None)
        else:
            terrain[pos] = kind
            deposits.pop(pos, None)

    for base in view["bases"]:
        MEM["bases"][base["baseId"]] = {"team": base["teamId"], "pos": _pos(base["position"])}


def _remember_actions(c, actions: ActionsByEntity) -> None:
    for u in c.workers:
        st = MEM["workers"].get(u["unitId"])
        if st is None:
            continue
        inv = _inventory(u)
        st["last"] = actions.get(u["unitId"], PASS)["type"] if _ready(c, u) else None
        st["last_pos"] = _pos(u["position"])
        st["last_load"] = inv["gold"] + inv["iron"]


# ---------------------------------------------------------------------------
# Per-turn context
# ---------------------------------------------------------------------------

class Ctx:
    pass


def _build_context(view: MarsView, team: str, turn: int, max_turns: int) -> Ctx:
    c = Ctx()
    c.view, c.team, c.turn = view, team, turn
    c.turns_left = max_turns - turn
    c.w, c.h = view["map"]["width"], view["map"]["height"]

    c.workers = [u for u in view["units"] if u["teamId"] == team]
    c.units_at = {}
    c.enemy_at = {}
    for u in view["units"]:
        pos = _pos(u["position"])
        c.units_at.setdefault(pos, []).append(u)
        if u["teamId"] != team:
            c.enemy_at[pos] = c.enemy_at.get(pos, 0) + 1

    bases = list(MEM["bases"].values())
    c.my_bases = [b["pos"] for b in bases if b["team"] == team]
    c.enemy_bases = [b["pos"] for b in bases if b["team"] != team]
    c.enemy_base_set = set(c.enemy_bases)
    c.base_set = set(c.my_bases) | c.enemy_base_set

    resources = view.get("teamResources") or {}
    mine = resources.get(team) or (next(iter(resources.values())) if len(resources) == 1 else {})
    c.bank_iron = mine.get("iron", 0)
    c.bank_gold = mine.get("gold", 0)
    c.bank_left = c.bank_iron

    area = view.get("highGoldArea") or {}
    area_w, area_h = area.get("width", 16), area.get("height", 16)
    x0, y0 = (c.w - area_w) // 2, (c.h - area_h) // 2
    c.center = (x0, y0, x0 + area_w - 1, y0 + area_h - 1)
    c.center_pt = (x0 + area_w // 2, y0 + area_h // 2)

    if MEM["home"] is None and c.my_bases:
        MEM["home"] = c.my_bases[0]

    c.cost = _cost_grid(c)
    c.base_dist = _dijkstra(c, c.my_bases)[0] if c.my_bases else [INF] * (c.w * c.h)

    # Other workers path around a builder's site so they don't block the build.
    builder_sites = {
        _idx(c, st["site"]): uid for uid, st in MEM["workers"].items()
        if st["role"] == "builder" and st.get("site")
    }
    c.paths = {}
    for u in c.workers:
        if _ready(c, u):
            blocked = {i for i, uid in builder_sites.items() if uid != u["unitId"]}
            c.paths[u["unitId"]] = _dijkstra(c, [_pos(u["position"])], blocked)

    c.hidden = [i for i in range(c.w * c.h) if (i % c.w, i // c.w) not in MEM["terrain"]]
    c.reserved = {}
    c.explore_picks = []
    return c


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _plan(c: Ctx, actions: ActionsByEntity) -> None:
    _assign_roles(c)
    _check_pending_build(c)
    _update_center_status(c)
    _check_blocked_mines(c)

    ready = []
    for u in c.workers:
        if _ready(c, u):
            ready.append(u)
        else:
            actions[u["unitId"]] = PASS

    free = []
    for u in ready:
        action = _endgame_action(c, u)
        if action:
            actions[u["unitId"]] = action
        else:
            free.append(u)

    _promote_builder(c, free)
    for u in free:
        if _state(u)["role"] == "builder":
            _safe(c, u, actions, _builder_action)
            if _state(u)["role"] != "builder" and actions[u["unitId"]] == PASS:
                del actions[u["unitId"]]  # gave up on the site; act normally this turn

    free = [u for u in free if u["unitId"] not in actions]
    spawner = _choose_spawner(c, free)
    if spawner:
        actions[spawner["unitId"]] = {"type": "spawn"}
    _assign_iron_worker(c, [u for u in free if u["unitId"] not in actions])

    gold_workers = []
    for u in free:
        if u["unitId"] in actions:
            continue
        role = _state(u)["role"]
        if role == "iron":
            _safe(c, u, actions, _iron_action)
        elif role == "center":
            _safe(c, u, actions, _center_action)
        if _state(u)["role"] == "gold" and actions.get(u["unitId"]) in (None, PASS):
            actions.pop(u["unitId"], None)
            gold_workers.append(u)

    _gold_actions(c, gold_workers, actions)


def _safe(c: Ctx, u, actions: ActionsByEntity, decide) -> None:
    try:
        action = decide(c, u)
    except Exception as error:
        print(f"T{c.turn:03d} worker {u['unitId']} error: {error!r}")
        action = None
    actions[u["unitId"]] = action or PASS


def _assign_roles(c: Ctx) -> None:
    first_sight = not MEM["order"]
    home = MEM["home"]
    march = home is not None and _cheb(home, c.center_pt) > CENTER_NEAR_HOME
    for u in c.workers:
        uid = u["unitId"]
        if uid in MEM["workers"]:
            continue
        k = len(MEM["order"])
        MEM["order"].append(uid)
        if first_sight and len(c.workers) > 1:
            role = "gold"  # restarted mid-match: skip the opening
        elif k == 1:
            role = "iron"
        elif k in (0, 2) and march:
            role = "center"
        else:
            role = "gold"
        MEM["workers"][uid] = {
            "role": role, "target": None, "site": None, "center": False,
            "last": None, "last_pos": None, "last_load": 0,
        }


def _center_plan_active(c: Ctx) -> bool:
    home = MEM["home"]
    march = home is not None and _cheb(home, c.center_pt) > CENTER_NEAR_HOME
    return (march and not MEM["center_base"] and not MEM["center_given_up"]
            and c.turn <= CENTER_GIVE_UP_TURN)


def _wait_for_center_iron(c: Ctx):
    """A center worker is idle because the base can't be afforded yet."""
    iron_coming = any(
        _inventory(u)["iron"] > 0 for u in c.workers
        if MEM["workers"].get(u["unitId"], {}).get("role") == "iron"
    )
    if not iron_coming:
        MEM["center_wait"] += 1
        if MEM["center_wait"] > CENTER_WAIT_LIMIT:
            MEM["center_given_up"] = True
    return PASS


def _update_center_status(c: Ctx) -> None:
    if any(_in_center(c, b, 2) for b in c.my_bases):
        MEM["center_base"] = True
    if _center_plan_active(c):
        return
    for st in MEM["workers"].values():
        if st["role"] == "center" or (st["role"] == "builder" and st["center"]):
            st["role"], st["site"], st["center"] = "gold", None, False


def _check_pending_build(c: Ctx) -> None:
    pending, MEM["pending"] = MEM["pending"], None
    if not pending:
        return
    site = pending["site"]
    st = MEM["workers"].get(pending["uid"])
    if site in c.my_bases:
        MEM["build_tries"].pop(site, None)
        if st:
            st["role"], st["site"], st["center"] = "gold", None, False
        return
    # A friendly worker standing on the site isn't the site's fault.
    if any(u["unitId"] != pending["uid"] for u in c.units_at.get(site, [])):
        return
    tries = MEM["build_tries"].get(site, 0) + 1
    MEM["build_tries"][site] = tries
    if tries >= 2:
        MEM["bad_sites"].add(site)
        if st:
            st["site"] = None


def _check_blocked_mines(c: Ctx) -> None:
    for u in c.workers:
        st = MEM["workers"].get(u["unitId"])
        pos = _pos(u["position"])
        if not st or st["last"] != "mine" or st["last_pos"] != pos:
            continue
        inv = _inventory(u)
        load = inv["gold"] + inv["iron"]
        if load == st["last_load"] and load < _capacity(u) and pos in MEM["deposits"]:
            MEM["blocked"][pos] = c.turn + BLOCKED_TURNS


# ---------------------------------------------------------------------------
# Endgame
# ---------------------------------------------------------------------------

def _endgame_action(c: Ctx, u):
    inv = _inventory(u)
    if inv["gold"] + inv["iron"] == 0:
        return None
    pos = _pos(u["position"])
    base = _nearest_base(c, u)
    if base is None:
        return None
    to_base = c.paths[u["unitId"]][0][_idx(c, base)]
    if c.turns_left > to_base + ENDGAME_MARGIN:
        return None
    if pos in c.my_bases:
        if inv["iron"] >= 3:
            return {"type": "trade", "from": "iron"}
        return {"type": "deposit"}
    return _move_to(c, u, base)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _spawn_cost(recruited: int) -> int:
    a, b = 5, 6
    for _ in range(recruited):
        a, b = b, a + b
    return a


def _choose_spawner(c: Ctx, free: list):
    count = len(c.workers)
    if not free or count >= MAX_WORKERS or c.turn >= STOP_SPAWN_TURN:
        return None
    building = any(st["role"] == "builder" for uid, st in MEM["workers"].items()
                   if uid in {u["unitId"] for u in c.workers})
    reserve = BASE_COST if (_center_plan_active(c) and count >= CENTER_RESERVE_WORKERS) or building else 0
    cost = _spawn_cost(count - 1)
    if c.bank_left - reserve < cost:
        return None

    order = MEM["order"]

    def rank(u):
        pos = _pos(u["position"])
        dep = MEM["deposits"].get(pos)
        on_gold = dep is not None and dep["type"] == "gold"
        index = order.index(u["unitId"]) if u["unitId"] in order else len(order)
        return (not on_gold, pos not in c.my_bases, index)

    c.bank_left -= cost
    return min(free, key=rank)


def _promote_builder(c: Ctx, free: list) -> None:
    ids = {u["unitId"] for u in c.workers}
    if any(st["role"] == "builder" for uid, st in MEM["workers"].items() if uid in ids):
        return

    if _center_plan_active(c):
        for u in free:
            st = _state(u)
            pos = _pos(u["position"])
            if st["role"] == "center" and _in_center(c, pos, 0):
                site = _best_site(c, pos, 4, False, u)
                if site:
                    st["role"], st["site"], st["center"] = "builder", site, True
                return
        return

    if c.turn >= STOP_BASE_TURN or c.bank_left < BASE_COST:
        return

    # The opening's center base fell through: build it as soon as we can afford it.
    if MEM["center_given_up"] and not MEM["center_base"]:
        golds = [u for u in free if _state(u)["role"] == "gold"]
        dist = lambda u: c.paths[u["unitId"]][0][_idx(c, c.center_pt)]
        u = min(golds, key=dist, default=None)
        if u is not None and dist(u) < c.turns_left - 100:
            site = _best_site(c, c.center_pt, 6, False, u)
            if site:
                st = _state(u)
                st["role"], st["site"], st["center"] = "builder", site, False
                return

    if len(c.workers) < WORKERS_BEFORE_EXTRA_BASES and c.turn < STOP_SPAWN_TURN:
        return  # save iron for workers first
    # Look at every known gold cluster far from our bases, not just current targets:
    # workers rarely pick far gold *because* there's no base near it yet.
    golds = [(p, d["amount"]) for p, d in MEM["deposits"].items()
             if d["type"] == "gold" and _can_mine(c, p)]
    clusters = []
    for p, _ in golds:
        if c.base_dist[_idx(c, p)] < NEW_BASE_MIN_DIST:
            continue
        nearby_gold = sum(a for q, a in golds if _cheb(p, q) <= 3)
        if nearby_gold >= NEW_BASE_MIN_GOLD:
            clusters.append((p, nearby_gold))

    builders = [u for u in free if _state(u)["role"] == "gold"]
    choices = []
    for p, nearby_gold in clusters:
        for u in builders:
            travel = c.paths[u["unitId"]][0][_idx(c, p)]
            if travel < c.turns_left - 50:
                choices.append((nearby_gold / (travel + 5), p, u))
    for _, p, u in sorted(choices, key=lambda choice: -choice[0]):
        site = _best_site(c, p, 3, True, u)
        if site:
            st = _state(u)
            st["role"], st["site"], st["center"] = "builder", site, False
            return


def _site_ok(c: Ctx, site, builder_id: str, need_far: bool) -> bool:
    x, y = site
    if not (0 <= x < c.w and 0 <= y < c.h):
        return False
    if MEM["terrain"].get(site) != "empty":
        return False
    if site in MEM["deposits"] or site in c.base_set or site in MEM["bad_sites"]:
        return False
    if any(_manhattan(site, e) <= 2 for e in c.enemy_bases):
        return False
    if any(u["unitId"] != builder_id for u in c.units_at.get(site, [])):
        return False
    if need_far and c.base_dist[_idx(c, site)] < NEW_BASE_MIN_DIST:
        return False
    return True


def _best_site(c: Ctx, near, radius: int, need_far: bool, builder):
    builder_pos = _pos(builder["position"])
    gold = [(p, d["amount"]) for p, d in MEM["deposits"].items() if d["type"] == "gold"]
    best, best_score = None, -INF
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            site = (near[0] + dx, near[1] + dy)
            if not _site_ok(c, site, builder["unitId"], need_far):
                continue
            score = 0.0
            for p, amount in gold:
                if _manhattan(site, p) <= 2:
                    score += amount
                elif _cheb(site, p) <= 3:
                    score += 0.5 * amount
            score -= 0.5 * _cheb(builder_pos, site)
            if score > best_score:
                best, best_score = site, score
    return best


def _builder_action(c: Ctx, u):
    st = _state(u)
    pos = _pos(u["position"])
    radius = 4 if st["center"] else 3
    need_far = not st["center"]

    site = st["site"]
    if site is None or not _site_ok(c, site, u["unitId"], need_far):
        site = _best_site(c, site or pos, radius, need_far, u)
        if site is None:
            st["role"], st["site"], st["center"] = "gold", None, False
            return None
        st["site"] = site

    if c.bank_left >= BASE_COST:
        if pos == site:
            c.bank_left -= BASE_COST
            MEM["pending"] = {"uid": u["unitId"], "site": site}
            return {"type": "build-base"}
        return _move_to(c, u, site)

    # Not enough iron yet: mine gold close to the site while we wait.
    option = _best_mine_option(c, u, near=site, radius=2, allow_return=False)
    if option:
        return _execute(c, u, option)
    if pos != site:
        return _move_to(c, u, site)
    return _wait_for_center_iron(c) if st["center"] else PASS


# ---------------------------------------------------------------------------
# Roles: iron and center
# ---------------------------------------------------------------------------

def _iron_goal(c: Ctx):
    """Bank iron we want for the next construction, or None if we're done building."""
    if _center_plan_active(c):
        return BASE_COST
    count = len(c.workers)
    can_spawn = count < MAX_WORKERS and c.turn < STOP_SPAWN_TURN
    if can_spawn and count < WORKERS_BEFORE_EXTRA_BASES:
        return _spawn_cost(count - 1)
    if c.turn < STOP_BASE_TURN:
        return BASE_COST
    return _spawn_cost(count - 1) if can_spawn else None


def _assign_iron_worker(c: Ctx, free: list) -> None:
    """When the bank is short of the next construction, send the worker closest to iron."""
    ids = {u["unitId"] for u in c.workers}
    if any(st["role"] == "iron" for uid, st in MEM["workers"].items() if uid in ids):
        return
    goal = _iron_goal(c)
    if goal is None or c.bank_left >= goal or _center_plan_active(c) or len(c.workers) < 2:
        return
    irons = [p for p, d in MEM["deposits"].items() if d["type"] == "iron" and _can_mine(c, p)]
    best, best_dist = None, INF
    for u in free:
        if _state(u)["role"] != "gold":
            continue
        dist = c.paths[u["unitId"]][0]
        for p in irons:
            if dist[_idx(c, p)] < best_dist:
                best, best_dist = u, dist[_idx(c, p)]
    if best is not None and best_dist < c.turns_left - 30:
        st = _state(best)
        st["role"], st["iron_since"], st["target"] = "iron", c.turn, None


def _iron_action(c: Ctx, u):
    st = _state(u)
    pos = _pos(u["position"])
    inv = _inventory(u)
    load = inv["gold"] + inv["iron"]
    goal = _iron_goal(c) or 0
    enough = c.bank_left + inv["iron"] >= goal  # bank_left: after this turn's spawn

    if load == 0 and (enough or c.turn - st.get("iron_since", 0) > IRON_ROLE_GIVE_UP_TURN):
        st["role"] = "gold"
        return None

    if pos in c.my_bases and load > 0:
        if enough:
            st["role"] = "gold"
        return {"type": "deposit"}

    if _capacity(u) - load > 0 and not enough:
        dep = MEM["deposits"].get(pos)
        if dep and dep["type"] == "iron" and _can_mine(c, pos):
            return {"type": "mine"}
        dist = c.paths[u["unitId"]][0]
        best, best_dist = None, INF
        for p, d in MEM["deposits"].items():
            if d["type"] != "iron" or not _can_mine(c, p):
                continue
            if d["amount"] - c.reserved.get(p, 0) <= 0:
                continue
            if dist[_idx(c, p)] < best_dist:
                best, best_dist = p, dist[_idx(c, p)]
        if best:
            c.reserved[best] = c.reserved.get(best, 0) + MEM["deposits"][best]["amount"]
            st["target"] = best
            return _move_to(c, u, best)

    if load > 0:
        base = _nearest_base(c, u)
        return _move_to(c, u, base) if base else None
    opening = st.get("iron_since", 0) == 0
    return _explore(c, u, home_radius=EXPLORE_HOME_RADIUS if opening else None)


def _center_action(c: Ctx, u):
    pos = _pos(u["position"])
    inv = _inventory(u)
    cap_left = _capacity(u) - inv["gold"] - inv["iron"]

    if not _in_center(c, pos, 0):
        if cap_left > 0:
            option = _best_mine_option(c, u, near=pos, radius=DETOUR_LIMIT,
                                       allow_return=False, gold_only=True)
            if option:
                return _execute(c, u, option)
        return _move_to(c, u, c.center_pt) or _explore(c, u)

    # Inside the center, waiting for the base to go up.
    if cap_left > 0:
        option = _best_mine_option(c, u, near=pos, radius=5, allow_return=False, gold_only=True)
        if option:
            return _execute(c, u, option)
        return _explore(c, u)
    return _wait_for_center_iron(c)


# ---------------------------------------------------------------------------
# Role: gold (greedy target assignment)
# ---------------------------------------------------------------------------

def _gold_actions(c: Ctx, workers: list, actions: ActionsByEntity) -> None:
    pending = list(workers)
    while pending:
        best_u, best_opt = None, None
        for u in pending:
            try:
                option = _choose_option(c, u)
            except Exception as error:
                print(f"T{c.turn:03d} worker {u['unitId']} error: {error!r}")
                option = (-INF, "idle", None, 0)
            if best_opt is None or option[0] > best_opt[0]:
                best_u, best_opt = u, option
        pending.remove(best_u)
        _safe(c, best_u, actions, lambda c, u: _execute(c, u, best_opt))


def _choose_option(c: Ctx, u):
    st = _state(u)
    inv = _inventory(u)
    load = inv["gold"] + inv["iron"]
    if load >= _capacity(u):
        return (INF, "return", None, 0)

    best = _best_mine_option(c, u)
    previous = st["target"]
    if best and previous and previous != best[2]:
        kept = _mine_value(c, u, previous)
        if kept and kept[0] >= best[0] / HYSTERESIS:
            best = kept

    choices = [best] if best else []
    if load > 0:
        base = _nearest_base(c, u)
        if base is not None:
            to_base = c.paths[u["unitId"]][0][_idx(c, base)]
            choices.append(((inv["gold"] + inv["iron"] * IRON_WEIGHT) / (to_base + 1), "return", None, 0))
    if not choices:
        return (-1, "explore", None, 0)
    return max(choices, key=lambda o: o[0])


def _want_iron(c: Ctx) -> bool:
    return len(c.workers) < MAX_WORKERS and c.turn < STOP_SPAWN_TURN


def _mine_value(c: Ctx, u, d, near=None, radius=None, allow_return=True, gold_only=False):
    dep = MEM["deposits"].get(d)
    if not dep or not _can_mine(c, d):
        return None
    if dep["type"] == "iron" and (gold_only or not _want_iron(c)):
        return None
    if near is not None and _cheb(near, d) > radius:
        return None

    inv = _inventory(u)
    cap_left = _capacity(u) - inv["gold"] - inv["iron"]
    avail = dep["amount"] // (c.enemy_at.get(d, 0) + 1) - c.reserved.get(d, 0)
    kg = min(avail, cap_left)
    if kg <= 0:
        return None

    travel = c.paths[u["unitId"]][0][_idx(c, d)]
    if travel == INF:
        return None
    back = c.base_dist[_idx(c, d)] if allow_return else 0
    if back == INF:
        return None
    turns = travel + kg + back + 1
    if turns > c.turns_left - ENDGAME_MARGIN:
        return None

    weight = 1 if dep["type"] == "gold" else IRON_WEIGHT
    value = (inv["gold"] + kg * weight) / turns
    if c.turn - dep["seen"] > STALE_TURNS:
        value *= STALE_FACTOR
    return (value, "mine", d, kg)


def _best_mine_option(c: Ctx, u, near=None, radius=None, allow_return=True, gold_only=False):
    best = None
    for d in MEM["deposits"]:
        option = _mine_value(c, u, d, near, radius, allow_return, gold_only)
        if option and (best is None or option[0] > best[0]):
            best = option
    return best


def _execute(c: Ctx, u, option):
    st = _state(u)
    _, kind, target, kg = option
    pos = _pos(u["position"])
    inv = _inventory(u)
    if st["role"] == "gold":
        st["target"] = target if kind == "mine" else None
    if kind == "mine":
        c.reserved[target] = c.reserved.get(target, 0) + kg

    if pos in c.my_bases and inv["gold"] + inv["iron"] > 0:
        return {"type": "deposit"}
    if kind == "return":
        base = _nearest_base(c, u)
        return _move_to(c, u, base) if base else PASS
    if kind == "mine":
        return {"type": "mine"} if pos == target else _move_to(c, u, target)
    return _explore(c, u)


def _explore(c: Ctx, u, home_radius=None):
    dist = c.paths[u["unitId"]][0]
    home = MEM["home"]
    best, best_score = None, INF
    for i in c.hidden:
        d = dist[i]
        if d == INF:
            continue
        p = (i % c.w, i // c.w)
        if home_radius and home and _cheb(p, home) > home_radius:
            continue
        if any(_cheb(p, q) <= 3 for q in c.explore_picks):
            continue
        score = d + 0.3 * _cheb(p, c.center_pt)
        if score < best_score:
            best, best_score = p, score
    if best is None:
        return _move_to(c, u, c.center_pt) or PASS
    c.explore_picks.append(best)
    return _move_to(c, u, best) or PASS


# ---------------------------------------------------------------------------
# Pathfinding and small helpers
# ---------------------------------------------------------------------------

def _cost_grid(c: Ctx) -> list:
    terrain = MEM["terrain"]
    cost = []
    for i in range(c.w * c.h):
        pos = (i % c.w, i // c.w)
        kind = terrain.get(pos)
        if pos in c.enemy_base_set or kind == "void":
            cost.append(INF)
        elif kind is None:
            cost.append(HIDDEN_COST)
        elif kind in ("mountain", "crater"):
            cost.append(OBSTACLE_COST)
        else:
            cost.append(1)
    return cost


def _dijkstra(c: Ctx, sources, blocked=frozenset()):
    """Turns from the nearest source to every tile, plus the previous tile on that route."""
    w, h, cost = c.w, c.h, c.cost
    dist = [INF] * (w * h)
    prev = [-1] * (w * h)
    queue = []
    for x, y in sources:
        dist[y * w + x] = 0
        queue.append((0, y * w + x))
    heapify(queue)
    while queue:
        d, i = heappop(queue)
        if d > dist[i]:
            continue
        x, y = i % w, i // w
        for dx, dy in STEPS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            j = ny * w + nx
            step = cost[j]
            if step == INF or j in blocked:
                continue
            nd = d + step
            if nd < dist[j]:
                dist[j] = nd
                prev[j] = i
                heappush(queue, (nd, j))
    return dist, prev


def _move_to(c: Ctx, u, target):
    if target is None:
        return None
    dist, prev = c.paths[u["unitId"]]
    start = _idx(c, _pos(u["position"]))
    goal = _idx(c, target)
    if start == goal or dist[goal] == INF:
        return None
    step = goal
    while prev[step] != start:
        step = prev[step]
        if step == -1:
            return None
    dx = step % c.w - start % c.w
    dy = step // c.w - start // c.w
    return {"type": "move", "direction": STEPS[(dx, dy)]}


def _nearest_base(c: Ctx, u):
    dist = c.paths[u["unitId"]][0]
    best = min(c.my_bases, key=lambda b: dist[_idx(c, b)], default=None)
    if best is None or dist[_idx(c, best)] == INF:
        return None
    return best


def _can_mine(c: Ctx, d) -> bool:
    if MEM["blocked"].get(d, -1) > c.turn:
        return False
    claimers = {b["team"] for b in MEM["bases"].values() if _manhattan(b["pos"], d) <= 2}
    return not claimers or c.team in claimers


def _in_center(c: Ctx, pos, margin: int) -> bool:
    x0, y0, x1, y1 = c.center
    return x0 - margin <= pos[0] <= x1 + margin and y0 - margin <= pos[1] <= y1 + margin


def _state(u) -> dict:
    return MEM["workers"][u["unitId"]]


def _ready(c: Ctx, u) -> bool:
    return c.turn >= u.get("readyOnTurn", 0)


def _inventory(u) -> dict:
    inv = u.get("inventory") or {}
    return {"gold": inv.get("gold", 0), "iron": inv.get("iron", 0)}


def _capacity(u) -> int:
    return u.get("inventoryCapacity", 8)


def _pos(point) -> tuple:
    return (point["x"], point["y"])


def _idx(c: Ctx, pos) -> int:
    return pos[1] * c.w + pos[0]


def _cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _log_turn(c: Ctx, actions: ActionsByEntity, ms: float) -> None:
    scores = c.view.get("scores") or {}
    ours = scores.get(c.team, c.bank_gold)
    rank = 1 + sum(1 for s in scores.values() if s > ours)
    parts = []
    for u in c.workers:
        st = MEM["workers"].get(u["unitId"], {})
        action = actions.get(u["unitId"], PASS)
        what = action["type"] + (" " + action["direction"] if action["type"] == "move" else "")
        parts.append(f"{u['unitId'][-4:]} {st.get('role', '?')} {what}")
    print(f"T{c.turn:03d} bank g={c.bank_gold} i={c.bank_iron} | rank {rank}/{max(len(scores), 1)} | "
          + " | ".join(parts) + f" | {ms:.0f}ms")
    for notice in (c.view.get("warnings") or []) + (c.view.get("info") or []):
        print(f"T{c.turn:03d} notice: {notice}")


# Game helpers. Feel free to use, discard, or modify.
# Fill in the strategy section above.

def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _cell(grid: CoordinateGrid | GameMap, point: Coordinate):
    x, y = point["x"], point["y"]
    if not _is_integer(x) or not _is_integer(y):
        return None
    if not (0 <= x < grid["width"] and 0 <= y < grid["height"]):
        return None

    row_start = y * grid["width"]
    return grid["cells"][row_start + x]


def is_claimed(state: TerritoryState | MarsView, coordinate: Coordinate) -> list[str]:
    """Return all known claiming teams, including overlapping claims."""
    teams: list[str] = []

    for base in state["bases"]:
        position = base["position"]
        distance = abs(position["x"] - coordinate["x"]) + abs(position["y"] - coordinate["y"])
        if distance <= 2 and base["teamId"] not in teams:
            teams.append(base["teamId"])

    return teams


def is_base(state: TerritoryState | MarsView, coordinate: Coordinate) -> Optional[str]:
    for base in state["bases"]:
        if base["position"] == coordinate:
            return base["teamId"]
    return None


def is_ore(state: OreState | MarsView, coordinate: Coordinate) -> Optional[Resource]:
    cell = _cell(state["map"], coordinate)
    if cell and cell["type"] in ("gold", "iron") and cell.get("amount", 0) > 0:
        return cell["type"]
    return None


def can_move_onto(
    state: MovementState | MarsView,
    coordinate: Coordinate,
    team_id: Optional[str] = None,
) -> bool:
    """Unknown terrain and enemy bases are excluded; workers do not block."""
    cell = _cell(state["map"], coordinate)
    if cell is None or cell["type"] in ("hidden", "void"):
        return False

    base_owner = is_base(state, coordinate)
    return base_owner is None or base_owner == team_id


def path_to(
    grid: CoordinateGrid | GameMap,
    start: Coordinate,
    end: Coordinate,
    *,
    bases: Sequence[BaseLocation] = (),
    team_id: Optional[str] = None,
    obstacle_cost: int = 3,
) -> Optional[list[Coordinate]]:
    """Cost until ready at destination; start must be ready; None if unreachable."""
    if not _is_integer(obstacle_cost) or obstacle_cost < 1:
        return None

    state: MovementState = {"map": grid, "bases": bases}
    if not can_move_onto(state, start, team_id) or not can_move_onto(state, end, team_id):
        return None

    origin = (start["x"], start["y"])
    target = (end["x"], end["y"])
    distances = {origin: 0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0, origin)]

    while queue:
        distance, position = heappop(queue)
        if distance != distances[position]:
            continue
        if position == target:
            break

        x, y = position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if not dx and not dy:
                    continue
                point: Coordinate = {"x": x + dx, "y": y + dy}
                if not can_move_onto(state, point, team_id):
                    continue

                neighbor = (point["x"], point["y"])
                cell = _cell(grid, point)
                movement_cost = obstacle_cost if cell and cell["type"] in ("mountain", "crater") else 1
                candidate = distance + movement_cost
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = position
                    heappush(queue, (candidate, neighbor))

    if target not in distances:
        return None

    path: list[Coordinate] = []
    position = target
    while position != origin:
        path.append({"x": position[0], "y": position[1]})
        position = previous[position]

    return list(reversed(path))


if __name__ == "__main__":
    print("Edit calculate_actions to choose your workers' actions.")
    if "LUCID_API_BASE_URL" not in os.environ:
        print(f"LUCID_API_BASE_URL is not set; using {API_BASE_URL}.")

    try:
        BotClient(API_KEY, API_BASE_URL).connect_and_run_bot(calculate_actions)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(error)
        raise SystemExit(1)
