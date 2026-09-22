"""Grid Runner env — novel game for AMD Academy Challenge.

Stages:
  S1: 9x9 open, 1 box / 1 plate, no patroller, no lava, full vision (starter)
  S2: 13x13 HARD — maze ~20%, lava 10-14, 3 boxes (1 decoy), 2 plates ORDERED,
      2 crossing patrollers, fog 5x5, 150 steps, 2 batteries off-path
  S3: 21x21 HARDER — maze ~25%, lava 25+, 4 boxes (1 decoy), 3 plates ORDERED,
      4 patrollers, fog 5x5, 300 steps, 3 batteries + key pickup

Rules:
  - Box push Sokoban-style (push into empty or plate only).
  - Plates must be covered IN ORDER by NON-DECOY boxes. Wrong order / decoy = no credit.
  - Patroller touch or lava = death. Door needs plates (+ key on S3) then step onto door.
  - Fog 5x5 returned to learner; full state available for expert / scoring.

Stdlib only — no torch needed for sim (keeps 3hr/day quota fast).
"""
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field

# cell codes for rendering / full state
EMPTY, WALL, LAVA, BATTERY, PLATE, DOOR, KEY = 0, 1, 2, 3, 4, 5, 6

ACTIONS = ("up", "down", "left", "right", "wait")
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "wait": (0, 0)}


@dataclass
class Patroller:
    loop: list  # list of (x, y)
    idx: int = 0
    tick: int = 0
    speed: int = 2  # move every `speed` env steps

    def pos(self):
        return tuple(self.loop[self.idx])

    def step(self):
        self.tick += 1
        if self.tick % self.speed == 0:
            self.idx = (self.idx + 1) % len(self.loop)


@dataclass
class GridConfig:
    name: str = "S1"
    w: int = 9
    h: int = 9
    n_boxes: int = 1
    n_plates: int = 1
    n_decoys: int = 0
    ordered: bool = False
    n_patrollers: int = 0
    lava_count: int = 0
    wall_density: float = 0.0
    fog_size: int = 9  # S1 full vision; S2/S3 use 5
    max_steps: int = 100
    n_batteries: int = 0
    battery_value: int = 15
    need_key: bool = False


def stage_config(stage: str) -> GridConfig:
    if stage == "S1":
        return GridConfig(name="S1", w=9, h=9, n_boxes=1, n_plates=1,
                          fog_size=9, max_steps=100)
    if stage == "S2":
        return GridConfig(name="S2", w=13, h=13, n_boxes=3, n_plates=2, n_decoys=1,
                          ordered=True, n_patrollers=2, lava_count=12,
                          wall_density=0.20, fog_size=5, max_steps=150,
                          n_batteries=2, battery_value=15)
    if stage == "S3":
        return GridConfig(name="S3", w=21, h=21, n_boxes=4, n_plates=3, n_decoys=1,
                          ordered=True, n_patrollers=4, lava_count=26,
                          wall_density=0.25, fog_size=5, max_steps=300,
                          n_batteries=3, battery_value=20, need_key=True)
    raise ValueError(f"unknown stage {stage}")


class GridEnv:
    def __init__(self, stage="S1", seed=0):
        self.cfg = stage_config(stage)
        self.rng = random.Random(seed)
        self.seed = seed
        self._gen()

    # ---------- generation ----------
    def _gen(self):
        c = self.cfg
        for _attempt in range(30):
            walls = set()
            # border walls
            for x in range(c.w):
                walls.add((x, 0)); walls.add((x, c.h - 1))
            for y in range(c.h):
                walls.add((0, y)); walls.add((c.w - 1, y))
            # random interior walls
            for y in range(2, c.h - 2):
                for x in range(2, c.w - 2):
                    if self.rng.random() < c.wall_density:
                        walls.add((x, y))
            start = (1, 1)
            door = (c.w - 2, c.h - 2)
            walls.discard(start); walls.discard(door)
            # keep connectivity start->door (BFS ignoring boxes/lava)
            if not self._connected(walls, start, door, c):
                continue
            free = [(x, y) for y in range(1, c.h - 1) for x in range(1, c.w - 1)
                    if (x, y) not in walls and (x, y) != start and (x, y) != door]
            self.rng.shuffle(free)
            need = c.lava_count + c.n_boxes + c.n_plates + c.n_batteries + (1 if c.need_key else 0) + 2
            if len(free) < need:
                continue
            lava = set(free[:c.lava_count])
            rest = free[c.lava_count:]
            boxes = {}
            for i in range(c.n_boxes):
                boxes[rest[i]] = f"box{i}"
            decoy_ids = {f"box{i}" for i in range(c.n_boxes - c.n_decoys, c.n_boxes)} if c.n_decoys else set()
            plates = rest[c.n_boxes:c.n_boxes + c.n_plates]
            batt_pos = rest[c.n_boxes + c.n_plates:c.n_boxes + c.n_plates + c.n_batteries]
            key_pos = rest[c.n_boxes + c.n_plates + c.n_batteries] if c.need_key else None
            # ensure plates/boxes not on lava, path still exists
            self.walls = walls; self.lava = lava; self.boxes = boxes
            self.decoy_ids = decoy_ids; self.plates = plates
            self.batteries = {p: c.battery_value for p in batt_pos}
            self.key_pos = key_pos; self.has_key = False
            self.door = door; self.player = start
            self.steps = 0; self.steps_left = c.max_steps
            self.plates_done = 0  # how many in-order plates satisfied
            self.patrollers = self._make_patrollers(c)
            self.done = False; self.won = False; self.death = None
            # solvability: reject corner-deadlocked boxes + require push plan on S1
            if self._deadlocked():
                continue
            if c.name == "S1" and not self._has_push_plan():
                continue
            return
        raise RuntimeError("failed to generate solvable map in 30 tries")

    def _deadlocked(self):
        # box in playable corner (two walls adjacent) that is not already on a plate = dead
        for pos in self.boxes:
            if pos in self.plates:
                continue
            x, y = pos
            if ((x - 1, y) in self.walls or (x + 1, y) in self.walls) and \
               ((x, y - 1) in self.walls or (x, y + 1) in self.walls):
                return True
        return False

    def _has_push_plan(self, max_expand=20000):
        walls = self.walls; lava = self.lava
        w, h = self.cfg.w, self.cfg.h
        box = next(iter(self.boxes)); plate = self.plates[0]
        start_p = self.player
        q = deque([(start_p, box)]); seen = {(start_p, box)}
        n = 0
        while q and n < max_expand:
            n += 1
            p, b = q.popleft()
            if b == plate:
                # also need player able to reach door after (walls/lava only)
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                np = (p[0] + dx, p[1] + dy)
                if not (0 <= np[0] < w and 0 <= np[1] < h) or np in walls or np in lava:
                    continue
                nb = b
                if np == b:
                    nb = (b[0] + dx, b[1] + dy)
                    if not (0 <= nb[0] < w and 0 <= nb[1] < h) or nb in walls or nb in lava:
                        continue
                if (np, nb) in seen:
                    continue
                seen.add((np, nb)); q.append((np, nb))
        return False

    def _connected(self, walls, start, door, c):
        q = deque([start]); seen = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == door:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (x + dx, y + dy)
                if 0 <= n[0] < c.w and 0 <= n[1] < c.h and n not in walls and n not in seen:
                    seen.add(n); q.append(n)
        return False

    def _make_patrollers(self, c):
        pats = []
        if c.n_patrollers == 0:
            return pats
        # crossing loops: horizontal middle + vertical middle (+ extras for S3)
        loops = []
        my = c.h // 2
        loops.append([(x, my) for x in range(2, c.w - 2)])
        mx = c.w // 2
        loops.append([(mx, y) for y in range(2, c.h - 2)])
        if c.n_patrollers > 2:
            loops.append([(x, 2) for x in range(2, c.w - 2)] + [(c.w - 3, y) for y in range(2, c.h - 2)])
            loops.append([(x, c.h - 3) for x in range(c.w - 3, 1, -1)] + [(1, y) for y in range(c.h - 3, 1, -1)])
        for i in range(c.n_patrollers):
            loop = [(x, y) for (x, y) in loops[i % len(loops)] if (x, y) not in self.walls]
            if len(loop) < 2:
                loop = [(2, 2), (2, 3)]
            pats.append(Patroller(loop=loop, idx=i % len(loop)))
        return pats

    # ---------- core loop ----------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed); self.seed = seed
        self._gen()
        return self.fog_obs()

    def step(self, action):
        if self.done:
            return self.fog_obs(), 0.0, True, {"info": "already done"}
        dx, dy = DIRS.get(action, (0, 0))
        px, py = self.player
        nx, ny = px + dx, py + dy
        reward = -0.1
        # wall / bounds
        if (nx, ny) in self.walls or not (0 <= nx < self.cfg.w and 0 <= ny < self.cfg.h):
            nx, ny = px, py
        # box push
        if (nx, ny) in self.boxes:
            bx, by = nx + dx, ny + dy
            if (bx, by) in self.walls or (bx, by) in self.boxes or (bx, by) in self.lava:
                nx, ny = px, py  # blocked
            else:
                bid = self.boxes.pop((nx, ny))
                self.boxes[(bx, by)] = bid
                nx, ny = nx, ny
                reward -= 0.05
        self.player = (nx, ny)
        self.steps += 1
        self.steps_left -= 1
        # patrollers move
        for p in self.patrollers:
            p.step()
        # pickups
        if self.player in self.batteries:
            self.steps_left += self.batteries.pop(self.player)
            reward += 5.0
        if self.key_pos and self.player == self.key_pos:
            self.has_key = True
            reward += 10.0
        # hazards
        if self.player in self.lava:
            return self._end(False, "lava", reward - 50.0)
        for p in self.patrollers:
            if p.pos() == self.player:
                return self._end(False, "patroller", reward - 50.0)
        # plates in order
        self._update_plates()
        # win: on door + plates done (+ key if needed)
        if self.player == self.door:
            if self.plates_done >= len(self.plates) and (not self.cfg.need_key or self.has_key):
                return self._end(True, "exit", reward + 100.0)
        if self.steps_left <= 0:
            return self._end(False, "timeout", reward - 20.0)
        return self.fog_obs(), reward, False, self._info()

    def _update_plates(self):
        # count longest in-order prefix of plates covered by NON-decoy boxes
        covered = {pos: bid for pos, bid in self.boxes.items() if pos in self.plates}
        done = 0
        for plate in self.plates:
            bid = covered.get(plate)
            if bid is None or bid in self.decoy_ids:
                break
            done += 1
        if done > self.plates_done:
            self.plates_done = done

    def _end(self, won, reason, reward):
        self.done = True; self.won = won; self.death = None if won else reason
        return self.fog_obs(), reward, True, self._info(reason)

    def _info(self, reason=None):
        return {"won": self.won, "reason": reason or self.death,
                "steps": self.steps, "plates_done": self.plates_done,
                "steps_left": self.steps_left, "stage": self.cfg.name}

    # ---------- observations ----------
    def fog_obs(self):
        c = self.cfg
        r = c.fog_size // 2
        px, py = self.player
        grid = []
        for y in range(py - r, py + r + 1):
            row = []
            for x in range(px - r, px + r + 1):
                row.append(self._cell_at(x, y))
            grid.append(row)
        return {"pos": self.player, "fog": grid, "fog_size": c.fog_size,
                "door_open": self.plates_done >= len(self.plates),
                "plates_done": self.plates_done, "steps_left": self.steps_left,
                "has_key": self.has_key, "stage": c.name}

    def _cell_at(self, x, y):
        if not (0 <= x < self.cfg.w and 0 <= y < self.cfg.h):
            return WALL
        if (x, y) in self.walls:
            return WALL
        if (x, y) in self.lava:
            return LAVA
        if (x, y) in self.batteries:
            return BATTERY
        if (x, y) in self.plates:
            return PLATE
        if (x, y) == self.door:
            return DOOR
        if self.key_pos and (x, y) == self.key_pos and not self.has_key:
            return KEY
        return EMPTY

    def full_state(self):
        return {"w": self.cfg.w, "h": self.cfg.h, "walls": sorted(self.walls),
                "lava": sorted(self.lava), "boxes": dict(self.boxes),
                "decoys": sorted(self.decoy_ids), "plates": list(self.plates),
                "door": self.door, "player": self.player, "batteries": dict(self.batteries),
                "key": self.key_pos, "has_key": self.has_key,
                "patrollers": [p.pos() for p in self.patrollers]}

    def render_ascii(self):
        sym = {EMPTY: ".", WALL: "#", LAVA: "~", BATTERY: "B", PLATE: "o", DOOR: "E", KEY: "K"}
        lines = []
        boxpos = set(self.boxes)
        patpos = {p.pos() for p in self.patrollers}
        for y in range(self.cfg.h):
            row = ""
            for x in range(self.cfg.w):
                if (x, y) == self.player:
                    row += "P"
                elif (x, y) in patpos:
                    row += "X"
                elif (x, y) in boxpos:
                    row += "D" if self.boxes[(x, y)] in self.decoy_ids else "b"
                else:
                    row += sym[self._cell_at(x, y)]
            lines.append(row)
        return "\n".join(lines)
