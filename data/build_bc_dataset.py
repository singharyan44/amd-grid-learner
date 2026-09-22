"""Build BC dataset from demo jsonl + BFS expert traces. Stdlib.

Usage:
  python data/build_bc_dataset.py --demos data/demos.jsonl --out data/bc_dataset.json
  python data/build_bc_dataset.py --expert --stage S1 --episodes 500 --out data/bc_expert_s1.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.grid_env import GridEnv
from agents.agents import heuristic_action
from agents.llm_agent import obs_to_prompt


def expert_traces(stage, episodes, seed0=9000):
    out = []
    for i in range(episodes):
        env = GridEnv(stage=stage, seed=seed0 + i)
        trace = []
        while True:
            o = env.fog_obs()
            a = heuristic_action(env)
            obs, r, done, info = env.step(a)
            trace.append({"prompt": obs_to_prompt(o), "action": a})
            if done:
                break
        if env.won:
            out.append({"stage": stage, "seed": seed0 + i, "steps": len(trace), "trace": trace})
    return out


def from_demos(path):
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("won"):
                continue
            trace = [{"prompt": obs_to_prompt(s["obs"]), "action": s["action"]}
                     for s in rec["trace"]]
            out.append({"stage": rec["stage"], "seed": rec["seed"],
                        "steps": len(trace), "trace": trace})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=None)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", default="data/bc_dataset.json")
    a = ap.parse_args()
    data = []
    if a.demos and os.path.exists(a.demos):
        data += from_demos(a.demos)
    if a.expert:
        data += expert_traces(a.stage, a.episodes)
    pairs = sum(len(d["trace"]) for d in data)
    with open(a.out, "w") as f:
        json.dump(data, f)
    print(json.dumps({"episodes": len(data), "pairs": pairs, "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
