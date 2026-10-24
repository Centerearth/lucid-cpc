"""Head-to-head between two bot files in the local simulator.

Half the teams run bot A and half run bot B; every seed is replayed with sides swapped so map luck
cancels out. Each team loads its own copy of its bot file, so module-level memory isn't shared.
A bot file needs calculate_actions(gameState, identity). Pass weight overrides for strategy.py as
strategy.py:key=value,key=value. Use strategy.py rather than bot_template.py, which shares one planner.

Usage: python3 versus.py A B [--teams=16] [--seeds=4]
  e.g. python3 versus.py strategy.py bot_template-marissa.py --teams=16
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from statistics import mean, stdev

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import sim  # noqa: E402

_loaded = 0


def load_bot(spec):
    """Load a fresh module instance for one team, applying any WEIGHTS overrides."""
    global _loaded
    path, _, overrides = spec.partition(":")
    _loaded += 1
    module_spec = importlib.util.spec_from_file_location(f"bot_{_loaded}", os.path.join(ROOT, path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if hasattr(module, "LOG"):
        module.LOG = False
    for item in filter(None, overrides.split(",")):
        key, value = item.split("=")
        module.WEIGHTS[key] = float(value)
    return module


def game(job):
    spec_a, spec_b, seed, flip, teams = job
    s = sim.Sim(seed, teams)
    is_a = {t: (k % 2 == 0) != flip for k, t in enumerate(s.teams)}
    bots = {t: load_bot(spec_a if is_a[t] else spec_b) for t in s.teams}
    worst = {True: 0.0, False: 0.0}
    errors = {True: 0, False: 0}
    while s.turn < s.max_turns:
        actions = {}
        for t, bot in bots.items():
            state = {"match": {"matchId": f"sim-{seed}", "status": "active", "turn": s.turn,
                               "maxTurns": s.max_turns, "teams": []},
                     "serverTimeMs": 0, "view": s.view(t)}
            started = time.monotonic()
            try:
                actions[t] = bot.calculate_actions(state, {"teamId": t, "displayName": t})
            except Exception:
                # Matches the real client: a strategy error submits no actions that turn.
                errors[is_a[t]] += 1
                actions[t] = {}
            worst[is_a[t]] = max(worst[is_a[t]], time.monotonic() - started)
        s.step(actions)
    gold = {side: [s.bank[t]["gold"] for t in s.teams if is_a[t] == side] for side in (True, False)}
    carried = {side: sum(u["inv"]["gold"] for u in s.units.values() if is_a[u["teamId"]] == side)
               for side in (True, False)}
    return gold, carried, worst, errors


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    options = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    spec_a, spec_b = args[:2]
    teams = int(options.get("teams", 16))
    seeds = range(3000, 3000 + int(options.get("seeds", 4)))
    jobs = [(spec_a, spec_b, seed, flip, teams) for seed in seeds for flip in (False, True)]
    started = time.monotonic()
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        results = pool.map(game, jobs)
    diffs = [mean(gold[True]) - mean(gold[False]) for gold, _, _, _ in results]
    a = [g for gold, _, _, _ in results for g in gold[True]]
    b = [g for gold, _, _, _ in results for g in gold[False]]
    se = stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    print(f"{len(jobs)} games, {teams} teams, {time.monotonic() - started:.0f}s")
    for name, golds, side in ((spec_a, a, True), (spec_b, b, False)):
        print(f"  {name}: mean gold {mean(golds):.0f} (min {min(golds)}, max {max(golds)}), "
              f"gold left carried {sum(r[1][side] for r in results)}, "
              f"worst turn {max(r[2][side] for r in results) * 1000:.0f} ms, "
              f"strategy errors {sum(r[3][side] for r in results)}")
    print(f"  A - B: {mean(diffs):+.1f} ± {se:.1f} gold per team")


if __name__ == "__main__":
    main()
