"""Baselines: random + BFS/heuristic. Stdlib only."""
import random
from collections import deque


def random_action(rng):
    return rng.choice(["up", "down", "left", "right", "wait"])


def bfs_path(env, start, goal, ignore_boxes=True):
    """Shortest player-only path avoiding walls/lava (and boxes optionally)."""
    st = env.full_state()
    walls = set(st["walls"]); lava = set(st["lava"])
    boxes = set(st["boxes"]) if not ignore_boxes else set()
    w, h = st["w"], st["h"]
    q = deque([(start, [])]); seen = {start}
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path
        for a, (dx, dy) in (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))):
            n = (x + dx, y + dy)
            if not (0 <= n[0] < w and 0 <= n[1] < h):
                continue
            if n in walls or n in lava or n in boxes or n in seen:
                continue
            seen.add(n); q.append((n, path + [a]))
    return []


def heuristic_action(env):
    """M1 baseline: full-state BFS push solver for S1 (1 box), greedy for S2/S3."""
    st = env.full_state()
    if len(st["boxes"]) == 1 and len(st["plates"]) == 1:
        plan = bfs_push_plan(env)
        if plan:
            return plan[0]
    # fallback greedy: walk to nearest box, push toward nearest plate
    return greedy_push(env)


def bfs_push_plan(env, max_expand=20000):
    """BFS over (player, box) — finds push sequence to plate then door. S1-sized only."""
    st = env.full_state()
    walls = set(st["walls"]); lava = set(st["lava"])
    w, h = st["w"], st["h"]
    box = next(iter(st["boxes"]))
    plate = st["plates"][0]; door = st["door"]
    start = (st["player"], box, False)  # (player, box, box_on_plate)
    q = deque([(start, [])]); seen = {(st["player"], box, False)}
    expand = 0
    while q and expand < max_expand:
        (p, b, onplate), path = q.popleft()
        expand += 1
        if onplate and p == door:
            return path
        for a, (dx, dy) in (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))):
            np = (p[0] + dx, p[1] + dy)
            if not (0 <= np[0] < w and 0 <= np[1] < h) or np in walls or np in lava:
                continue
            nb, nonplate = b, onplate
            if np == b:  # push
                nb = (b[0] + dx, b[1] + dy)
                if not (0 <= nb[0] < w and 0 <= nb[1] < h) or nb in walls or nb in lava:
                    continue
                nonplate = onplate or (nb == plate)
            else:
                nonplate = onplate or (b == plate)
            key = (np, nb, nonplate)
            if key in seen:
                continue
            seen.add(key); q.append(((np, nb, nonplate), path + [a]))
    return []


def greedy_push(env):
    """S2/S3 fallback: step toward box, push to reduce box->plate distance."""
    import random as _r
    st = env.full_state()
    player = st["player"]
    boxes = [p for p in st["boxes"] if st["boxes"][p] not in set(st.get("decoys", []))] or list(st["boxes"])
    if not boxes:
        return _r.choice(["up", "down", "left", "right"])
    plate = st["plates"][0]
    # nearest box by Manhattan
    box = min(boxes, key=lambda b: abs(b[0] - player[0]) + abs(b[1] - player[1]))
    # if adjacent, try push that reduces box->plate dist
    for a, (dx, dy) in (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))):
        if (player[0] + dx, player[1] + dy) == box:
            nb = (box[0] + dx, box[1] + dy)
            walls = set(st["walls"]); lava = set(st["lava"])
            if nb in walls or nb in lava or nb in st["boxes"]:
                continue
            before = abs(box[0] - plate[0]) + abs(box[1] - plate[1])
            after = abs(nb[0] - plate[0]) + abs(nb[1] - plate[1])
            if after <= before:
                return a
    # else walk toward box
    path = bfs_path(env, player, box, ignore_boxes=False)
    if path:
        return path[0]
    return _r.choice(["up", "down", "left", "right"])
