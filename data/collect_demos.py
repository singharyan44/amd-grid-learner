"""Human demo collector — terminal WASD play, records traces for BC (M2).

Usage:
  python data/collect_demos.py --stage S1 --episodes 5 --out data/demos_s1.jsonl

Controls: w=up s=down a=left d=right space=wait, q=quit episode, r=reset.
Records per step: fog_obs + action + reward + done. Only winning episodes
are kept for BC by default (--keep all|wins).
Stdlib only.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.grid_env import GridEnv

KEYS = {"w": "up", "s": "down", "a": "left", "d": "right", " ": "wait"}


def play_episode(env):
    trace = []
    obs = env.fog_obs()
    print(env.render_ascii())
    print(f"steps_left={obs['steps_left']} plates={obs['plates_done']}")
    while True:
        try:
            k = input("move [w/a/s/d/space, r=reset, q=quit]: ").strip().lower()
        except EOFError:
            return None, True
        if k == "q":
            return None, True
        if k == "r":
            return None, False
        if k == "" :
            k = " "
        if k not in KEYS:
            print("use w/a/s/d/space")
            continue
        action = KEYS[k]
        obs_before = env.fog_obs()
        obs, reward, done, info = env.step(action)
        trace.append({"obs": obs_before, "action": action,
                      "reward": reward, "done": done})
        print(env.render_ascii())
        print(f"action={action} reward={reward:.1f} steps_left={obs['steps_left']} plates={obs['plates_done']}")
        if done:
            print("WON!" if info["won"] else f"lost: {info['reason']}")
            return trace, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default="data/demos.jsonl")
    ap.add_argument("--keep", default="wins", choices=["wins", "all"])
    ap.add_argument("--seed0", type=int, default=2000)
    a = ap.parse_args()
    kept, played = 0, 0
    with open(a.out, "a") as f:
        for i in range(a.episodes):
            env = GridEnv(stage=a.stage, seed=a.seed0 + i)
            print(f"=== episode {i+1}/{a.episodes} seed={a.seed0+i} stage={a.stage} ===")
            trace, quit_all = play_episode(env)
            if quit_all:
                break
            if trace is None:
                continue
            played += 1
            won = env.won
            if a.keep == "wins" and not won:
                print("dropped (loss) — only wins kept")
                continue
            rec = {"stage": a.stage, "seed": a.seed0 + i, "won": won,
                   "steps": len(trace), "trace": trace}
            f.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"played={played} kept={kept} -> {a.out}")


if __name__ == "__main__":
    main()
