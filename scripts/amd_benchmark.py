"""Bench: win-rate + sim throughput + (on ROCm) VRAM/latency. Stdlib for M1."""
import argparse, json, subprocess, sys, time
sys.path.insert(0, ".")
from train.parallel_runner import rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    from multiprocessing import Pool
    jobs = [(a.stage, "random", 5000 + i) for i in range(a.episodes)]
    with Pool(a.workers) as pool:
        res = pool.map(rollout, jobs)
    wins = sum(1 for r in res if r["won"])
    out = {"stage": a.stage, "win_rate": wins / len(res),
           "games_per_sec": len(res) / max(time.time() - t0, 1e-6)}
    # ROCm GPU info if present (no fail on CPU boxes)
    try:
        smi = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=5)
        out["rocm_smi"] = (smi.stdout or "")[:500]
    except Exception as e:
        out["rocm_smi"] = f"not-available: {e}"
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
