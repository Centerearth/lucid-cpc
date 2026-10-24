"""Random-search tuner for strategy.WEIGHTS using the local simulator.

Each candidate plays head-to-head against the default weights: half the teams each, and every
seed is replayed with sides swapped, so map luck cancels out. A candidate's score is the mean
per-game gold difference (candidate teams minus default teams). The best candidates get more games
on new seeds, and the last stage uses held-out seeds so the winner's reported edge is not inflated.

Usage: python3 tune.py [candidates] [output.json] [--teams=N] [--rng=N] [--quick]
"""
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402
from strategy import WEIGHTS, Planner  # noqa: E402

TEAMS = 6

# name: (low, high, integer?). The first 16-team run favored less exploring, longer trust in old
# sightings, and more stickiness; the rest cover the switches adopted from Marissa's bot.
SPACE = {
    "explore_value": (0.3, 4.0, False),
    "explore_center_pull": (0.0, 0.8, False),
    "explore_min_turns_left": (10, 120, True),
    "stale_turns": (30.0, 200.0, False),
    "unknown_cost": (1.0, 2.5, False),
    "contention_penalty": (0.0, 1.5, False),
    "stickiness": (1.0, 1.7, False),
    "base_min_savings": (3, 12, True),
    "base_cluster_min_gold": (8, 24, True),
    "base_cluster_min_dist": (3, 8, True),
    "save_workers_until": (2, 6, True),
    "iron_value_needed": (0.5, 2.5, False),
    "worker_gold_per_turn": (0.2, 0.6, False),
}


def sample(rng):
    return {name: rng.randint(low, high) if is_int else round(rng.uniform(low, high), 3)
            for name, (low, high, is_int) in SPACE.items()}


def game(job):
    index, weights, seed, flip, teams = job
    s = sim.Sim(seed, teams)
    is_candidate = {t: (k % 2 == 0) != flip for k, t in enumerate(s.teams)}
    planners = {t: Planner(t, "sim", sim.W, sim.H, weights if is_candidate[t] else None) for t in s.teams}
    while s.turn < s.max_turns:
        s.step({t: p.decide(s.view(t), s.max_turns) for t, p in planners.items()})
    ours = mean(s.bank[t]["gold"] for t in s.teams if is_candidate[t])
    theirs = mean(s.bank[t]["gold"] for t in s.teams if not is_candidate[t])
    return index, seed, ours - theirs


def summary(diffs):
    if not diffs:
        return "no games"
    se = stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    return f"{mean(diffs):+6.1f} ± {se:4.1f} over {len(diffs)} games"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    teams = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--teams=")), TEAMS)
    rng_seed = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--rng=")), 0)
    n = int(args[0]) if args else 160
    out = args[1] if len(args) > 1 else "tune_results.json"
    rng = random.Random(rng_seed)
    # Candidate 0 is the defaults, a control that should score about zero.
    candidates = [{}] + [sample(rng) for _ in range(n - 1)]
    diffs = {i: [] for i in range(n)}
    held_out = {i: [] for i in range(n)}
    # (seeds, how many candidates enter the stage, held out?)
    stages = ([(range(1000, 1001), n, False), (range(2000, 2001), 2, True)] if quick else
              [(range(1000, 1002), n, False), (range(1002, 1006), max(4, n // 5), False),
               (range(2000, 2004 if teams > 6 else 2008), 3, True)])
    alive = list(range(n))
    started = time.monotonic()
    with Pool(os.cpu_count()) as pool:
        for stage, (seeds, keep, is_held_out) in enumerate(stages, 1):
            alive = sorted(alive, key=lambda i: -mean(diffs[i]) if diffs[i] else 0)[:keep]
            if is_held_out and 0 not in alive:
                alive.append(0)
            jobs = [(i, candidates[i], seed, flip, teams) for i in alive for seed in seeds for flip in (False, True)]
            for done, (i, seed, diff) in enumerate(pool.imap_unordered(game, jobs), 1):
                (held_out if is_held_out else diffs)[i].append(diff)
                if done % 20 == 0 or done == len(jobs):
                    print(f"stage {stage}: {done}/{len(jobs)} games, {time.monotonic() - started:.0f}s", flush=True)
            ranked = sorted(alive, key=lambda i: -mean(held_out[i] if is_held_out else diffs[i]))
            for i in ranked[:5]:
                print(f"  candidate {i:3d}: search {summary(diffs[i])}; held out {summary(held_out[i])}", flush=True)
            with open(out, "w") as f:
                json.dump({"defaults": WEIGHTS, "stage": stage, "results": [
                    {"candidate": i, "weights": candidates[i], "search": diffs[i], "held_out": held_out[i]}
                    for i in ranked]}, f, indent=1)
    print(f"done in {time.monotonic() - started:.0f}s; results in {out}")


if __name__ == "__main__":
    main()
