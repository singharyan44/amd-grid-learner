"""Parallel rollout: N envs x M episodes, stdlib multiprocessing."""
import argparse, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from env.grid_env import GridEnv
from agents.agents import random_action, heuristic_action
import random


def rollout(args):
    stage, policy, seed = args
    env = GridEnv(stage=stage, seed=seed)
    rng = random.Random(seed)
    obs = env.fog_obs()
    total = 0.0
    steps = 0
    while True:
        a = random_action(rng) if policy == "random" else heuristic_action(env)
        obs, r, done, info = env.step(a)
        total += r; steps += 1
        if done:
            return {"won": info["won"], "reason": info["reason"],
                    "steps": steps, "reward": total, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--policy", default="random", choices=["random", "heuristic"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="data/baseline.json")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(a.stage, a.policy, 1000 + i) for i in range(a.episodes)]
    with Pool(a.workers) as pool:
        results = pool.map(rollout, jobs)
    wins = sum(1 for r in results if r["won"])
    dt = time.time() - t0
    summary = {"stage": a.stage, "policy": a.policy, "episodes": a.episodes,
               "wins": wins, "win_rate": wins / len(results),
               "games_per_sec": len(results) / max(dt, 1e-6), "seconds": dt}
    print(json.dumps(summary, indent=2))
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
