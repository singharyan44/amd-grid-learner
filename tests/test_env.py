"""M1 tests: rules, fog, door, decoy, patroller. Run: python -m pytest tests/ -q"""
from env.grid_env import GridEnv


def test_s1_gen():
    e = GridEnv(stage="S1", seed=1)
    assert e.cfg.w == 9 and not e.done


def test_wall_block():
    e = GridEnv(stage="S1", seed=1)
    e.player = (1, 1)
    obs, r, done, info = e.step("left")  # into border wall
    assert e.player == (1, 1) and not done


def test_fog_size_s2():
    e = GridEnv(stage="S2", seed=7)
    obs = e.fog_obs()
    assert len(obs["fog"]) == 5 and len(obs["fog"][0]) == 5


def test_battery_pickup():
    e = GridEnv(stage="S2", seed=7)
    # place battery under player path deterministically
    px, py = e.player
    e.batteries[(px + 1, py)] = 15
    e.walls.discard((px + 1, py))
    before = e.steps_left
    e.step("right")
    assert e.steps_left >= before  # picked up net of -1 step


def test_lava_death():
    e = GridEnv(stage="S1", seed=1)
    px, py = e.player
    e.lava.add((px + 1, py))
    _, _, done, info = e.step("right")
    assert done and info["reason"] == "lava"


def test_decoy_does_not_count():
    e = GridEnv(stage="S2", seed=11)
    # force: put decoy box on first plate
    plate = e.plates[0]
    decoy = next(iter(e.decoy_ids))
    # clear and place
    for pos, bid in list(e.boxes.items()):
        if bid == decoy:
            del e.boxes[pos]
    e.boxes[plate] = decoy
    e._update_plates()
    assert e.plates_done == 0
