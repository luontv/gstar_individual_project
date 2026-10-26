import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gym_cooking.play_test import MAP_SETTINGS
from agent.executor.low import EnvState
from agent.executor.high import OBJ_TO_GOODS_GS, OBJ_TO_GOODS_POT
from agent.mind.prompt_local import prep_chk_moves, prep_prompt_order, prep_prompt_map_s

MAPS = MAP_SETTINGS | {  # the 4 paper maps + 2 extra HLA layouts (training only)
    'new4': dict(level='new4'),
    'change': dict(level='change'),
}
AI, HUMAN = 0, 1
STEP_SECONDS = 0.4  # HLA runs its agent at 2.5 fps
MACRO_TIMEOUT_STEPS = 40


def env_state(env, idx):
    info = env.get_ai_info()
    return EnvState(world=info['world'], agents=info['sim_agents'], agent_idx=idx,
                    order=info['order_scheduler'], event_history=info['event_history'],
                    time=info['current_time'], chg_grid=info['chg_grid'])


def name_of(obj, table=OBJ_TO_GOODS_GS):
    return None if obj is None else table.get(obj.full_name, obj.full_name)


STAGES = ('Putout', 'Drop', 'Serve', 'Plate', 'Cook', 'Prepare', 'Chop')


# mostly the furthest-along step for a current order, sometimes a random one
def pick_macro(e, rng, greedy=0.7):
    feasible = [m for m, ok, *_ in prep_chk_moves(e) if ok]
    if not feasible:
        return None
    orders = [o['name'].replace('Plated ', '') for o in prep_prompt_order(e)]
    names = {o.split()[0] for o in orders}  # Alice, Bob, ...
    relevant = [m for m in feasible
                if m.split()[0] in ('Putout', 'Drop', 'Chop') or m.split()[1] in names]
    if relevant and rng.random() < greedy:
        for stage in STAGES:
            best = [m for m in relevant if m.startswith(stage)]
            if best:
                return rng.choice(best)
    return rng.choice(relevant or feasible)


def snapshot(env, game_map, e, hist):
    res = e.get_all_grid_info()
    pots, items = [], []
    for g in res:
        if not g['rch']:
            continue
        if g['gs'].name == 'Pot':
            obj = g['obj']
            status = None if obj is None else next(
                (s for s in ('Cooking', 'Cooked', 'Fire', 'Charred') if s in obj.full_name), 'Other')
            pots.append({'pos': g['pos'], 'soup': name_of(obj, OBJ_TO_GOODS_POT), 'status': status})
        elif g['gs'].name in ('Counter', 'Cutboard') and g['obj'] is not None:
            items.append({'pos': g['pos'], 'on': g['gs'].name, 'item': name_of(g['obj'])})
    ai, human = e.agents[AI], e.agents[HUMAN]
    return {
        'map': game_map,
        'time': round(env.current_time, 1),
        'orders': prep_prompt_order(e),
        'pots': pots,
        'items': items,
        'ai': {'pos': ai.location, 'hold': name_of(ai.holding)},
        'human': {'pos': human.location, 'hold': name_of(human.holding)},
        'ai_hist': hist[AI][-5:],
        'human_hist': hist[HUMAN][-5:],
        'feasible': [m for m, ok, *_ in prep_chk_moves(e) if ok],
        'map_text': prep_prompt_map_s(e),
    }
