import argparse
import json
import os
import random
from copy import deepcopy

from askoract import clarify as K
from askoract.game import AI, HUMAN, MACRO_TIMEOUT_STEPS, MAPS, STEP_SECONDS, env_state, pick_macro, snapshot
import pygame
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.misc.game.game import Game
from gym_cooking.utils.core import Object, Plate, Tomato, Lettuce, Onion
from gym_cooking.recipe_planner.utils import Cooked
from agent.executor.high import HighTask, OBJ_TO_GOODS_GS
from agent.mind.prompt_local import MOVE_TO_HT, prep_prompt_order


RECIPE = {soup: sorted(ings) for soup, ings in K.RECIPE.items()}  # sorted so objects are always placed in the same order
FOOD = {'Tomato': Tomato, 'Lettuce': Lettuce, 'Onion': Onion}
CHOPPED, COOKED = 2, 4  # indices in the food state sequence
STAGES_INJECT = ('chopped', 'prepared', 'cooked', 'plated')


def food(name, state, t):
    f = FOOD[name](state)
    if state == COOKED:  # the burn timer needs a start time
        f.state = Cooked(name, start_time=t)
    return f


# copy of the kitchen with two soups at the same stage (chopped / prepared / cooked / plated),
# so soup-level commands become ambiguous
def busy_kitchen(env, rng):
    env = deepcopy(env)
    e = env_state(env, AI)
    ordered = list(dict.fromkeys(o['name'].split()[1] for o in prep_prompt_order(e)))
    soups = (ordered + rng.sample([x for x in RECIPE if x not in ordered], 4 - len(ordered)))[:max(2, len(ordered))]
    soups = rng.sample(soups, 2)
    stage = rng.choice(STAGES_INJECT)
    if rng.random() < 0.5:  # make one of the two urgent so urgency and intent can disagree
        urgent = rng.choice(soups)
        sched = env.order_scheduler
        sched.current_orders = [
            (o, lim * 0.15 if OBJ_TO_GOODS_GS[o.full_name].split()[1] == urgent else max(rest, lim * 0.8), lim, b)
            for o, rest, lim, b in sched.current_orders]
    free = [g['gs'] for g in e.get_all_grid_info() if g['obj'] is None and g['rch']]
    counters = [g for g in free if g.name == 'Counter']
    pots = [g for g in free if g.name == 'Pot']
    rng.shuffle(counters)
    t = env.current_time
    if stage == 'chopped':
        loose = sorted({i for x in soups for i in RECIPE[x]})
        placed = [(counters, [food(i, CHOPPED, t)]) for i in loose]
    elif stage == 'prepared':
        placed = [(counters, [food(i, CHOPPED, t) for i in RECIPE[x]]) for x in soups]
    elif stage == 'cooked':
        placed = [(pots, [food(i, COOKED, t) for i in RECIPE[x]]) for x in soups]
    else:
        placed = [(counters, [food(i, COOKED, t) for i in RECIPE[x]] + [Plate()]) for x in soups]
    for where, contents in placed:
        if not where:
            return None, None
        gs = where.pop()
        obj = Object(gs.location, contents)
        gs.acquire(obj)
        env.world.insert(obj)
    return env, stage


def render(env, path):
    g = Game(env)
    g.on_init()
    g.on_render()
    pygame.image.save(g.screen, path)


def play(game_map, seed, every, max_time, p_busy, render_dir=None):
    rng = random.Random(seed)
    random.seed(seed)  # order schedule uses the global RNG
    env = OvercookedEnvironment(MapSetting(**MAPS[game_map], max_num_timesteps=max_time))
    env.reset()
    tasks, started, hist, out = [None, None], [0, 0], [[], []], []
    for step in range(10 ** 6):
        actions = {}
        for idx in (AI, HUMAN):
            e = env_state(env, idx)
            if tasks[idx] is None:
                m = pick_macro(e, rng)
                tasks[idx] = deepcopy(MOVE_TO_HT[m]) if m else None
                started[idx] = step
                if m:
                    hist[idx].append(m)
            move = (0, 0)
            if tasks[idx] is not None:
                state, move, _ = tasks[idx](e)
                # executors can wait forever (e.g. for a free cutboard), so cap each macro
                if state != HighTask.Working or step - started[idx] >= MACRO_TIMEOUT_STEPS:
                    tasks[idx] = None
            actions[env.sim_agents[idx].name] = move
        _, _, done, _ = env.step(actions, passed_time=STEP_SECONDS)
        if done:
            return out
        if step % every == every - 1:
            shots = [(env, None)]
            if rng.random() < p_busy:
                busy, stage = busy_kitchen(env, rng)
                if busy is not None:
                    shots.append((busy, stage))
            for e_, stage in shots:
                rec = snapshot(e_, game_map, env_state(e_, AI), hist) | {'injected': stage}
                if render_dir:
                    rec['image'] = f'{game_map}-{seed}-{step}{"-" + stage if stage else ""}.png'
                    render(e_, os.path.join(render_dir, rec['image']))
                out.append(rec)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=5, help='games per map')
    p.add_argument('--every', type=int, default=10, help='steps between snapshots')
    p.add_argument('--max-time', type=int, default=200, help='game length in seconds')
    p.add_argument('--busy', type=float, default=0.5, help='chance of adding a busy-kitchen copy per snapshot')
    p.add_argument('--render', default=None, help='folder to save a screenshot of every snapshot')
    p.add_argument('--maps', default=','.join(MAP_SETTINGS), help='comma-separated map names')
    p.add_argument('--out', default='data/states.jsonl')
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    if a.render:
        os.makedirs(a.render, exist_ok=True)
    n = 0
    with open(a.out, 'w') as f:
        for game_map in a.maps.split(','):
            for g in range(a.games):
                for s in play(game_map, seed=1000 * g + len(game_map), every=a.every, max_time=a.max_time,
                              p_busy=a.busy, render_dir=a.render):
                    s['game'] = f'{game_map}-{g}'
                    f.write(json.dumps(s) + '\n')
                    n += 1
    print(f'wrote {n} snapshots to {a.out}')
