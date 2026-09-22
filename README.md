# AMD Grid Game Learner
Fresh solo build for **Lablab x AMD AI Academy Challenge** (Sept 1 – Dec 1, 2026). Separate from UNSEEN.

Novel grid game + learner from others (imitation) + from itself (self-play + RL).

## M1 quickstart
```powershell
python -m pytest tests/ -q
python train/parallel_runner.py --stage S1 --policy random --episodes 200 --workers 8
python train/parallel_runner.py --stage S1 --policy heuristic --episodes 200 --workers 8
python scripts/amd_benchmark.py --stage S1 --episodes 200
```
## Stages
- S1 9x9 open (starter), S2 13x13 HARD (maze/lava/decoy/ordered/patrollers/fog 5x5), S3 21x21 HARDER (+key, 4 patrollers)
