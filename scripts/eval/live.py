import argparse
import json
import random
import time
from copy import deepcopy

import numpy as np

from askoract import clarify as K
from askoract.backends import API
from askoract.game import AI, HUMAN, MACRO_TIMEOUT_STEPS, MAPS, STEP_SECONDS, env_state, pick_macro, snapshot
from askoract.policies import never_ask_best, oracle_rule, text_rule
from askoract.prompt import acted_obj, messages, parse, question
from scripts.data.build_new_wording import NEW
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from agent.executor.high import HighTask
from agent.mind.prompt_local import MOVE_TO_HT

FIRST_GAME = 5  # games 0-4 gave the training/test moments, so these are new
LIVE_MAPS = ('ring', 'bottleneck', 'partition', 'quick')  # HLA's four maps; partition is never trained on
RULES = {'never_ask': never_ask_best, 'keyword': text_rule, 'perfect_rule': oracle_rule}


# a command for the intended step, using any reference type that can point to it
def command(s, rng, intended):
    verb, gold = K.split(intended)
    refs = [('explicit', [gold])]
    for ref in ('bare', 'other', 'urgent', 'holding', 'near'):
        c = K.candidates(verb, ref, s) if (verb, ref) in K.TEMPLATES else None
        if c and gold in c:
            refs.append((ref, c))
    if s['ai_hist'] and s['ai_hist'][-1] == intended:
        refs.append(('again', [gold]))
    ref, cands = rng.choice(refs)
    key = (None, 'again') if ref == 'again' else (verb, ref)
    return K.make(0, s, verb, ref, rng.randrange(len(K.TEMPLATES[key])), cands, gold)


# returns (object to act on or None, asked, seconds the AI waits, objects that count as right, latencies ms)
def decide(policy, e, s, rng, gen, reply_delay, latency_scale=1.0):
    verb, gold, cands = e['verb'], e['gold'], e['cands']
    if policy == 'mind_reader':
        return gold, False, 0.0, [gold], []
    if policy in RULES:
        kind, x = RULES[policy](e, s, rng)
        if kind != 'ask':
            return x, False, 0.0, [gold], []
        q = {'options': x}
        _, val, ok = K.reply(q, gold, cands, verb, rng)
        return K.settle(q, val, cands, verb, s), True, reply_delay, ok, []
    t = time.perf_counter()
    out = parse(gen([messages(s, e['command'])])[0])
    l1 = 1000 * (time.perf_counter() - t)
    if out.get('mode') != 'ask':
        return acted_obj(out, verb), False, latency_scale * l1 / 1000, [gold], [l1]
    q = question(out)
    text, _, ok = K.reply(q, gold, cands, verb, rng)
    asked = out.get('question') or K.say(q['options'] or ['it'], verb)
    t = time.perf_counter()
    out = parse(gen([messages(s, e['command'], asked, text)])[0])
    l2 = 1000 * (time.perf_counter() - t)
    return acted_obj(out, verb), True, latency_scale * (l1 + l2) / 1000 + reply_delay, ok, [l1, l2]


def play(game_map, g, policy, gen, args):
    seed = 1000 * g + len(game_map)
    rng = random.Random(seed)
    random.seed(seed)  # order schedule uses the global RNG
    np.random.seed(seed)  # so do the testbed's executors
    env = OvercookedEnvironment(MapSetting(**MAPS[game_map], max_num_timesteps=args.max_time))
    env.reset()
    tasks, started, hist, log = [None, None], [0, 0], [[], []], []
    free_at, pending = args.gap, None  # game time when the AI may start; macro decided but not started yet

    def start(idx, m, step):
        tasks[idx], started[idx] = deepcopy(MOVE_TO_HT[m]), step
        hist[idx].append(m)

    for step in range(10 ** 6):
        now = step * STEP_SECONDS
        if tasks[HUMAN] is None:
            m = pick_macro(env_state(env, HUMAN), rng)
            if m:
                start(HUMAN, m, step)
        if policy != 'idle' and tasks[AI] is None and now >= free_at:
            if pending:
                start(AI, pending, step)
                pending = None
            else:
                e_ai = env_state(env, AI)
                intended = pick_macro(e_ai, rng)
                if intended and K.split(intended)[0] not in K.VERBS:  # Putout / Drop: no command
                    start(AI, intended, step)
                elif intended:
                    s = snapshot(env, game_map, e_ai, hist) | {'game': f'{game_map}-{g}'}
                    e = command(s, rng, intended)
                    obj, asked, wait, ok, lat = decide(policy, e, s, rng, gen, args.reply_delay, args.latency_scale)
                    m = K.macro(e['verb'], obj) if obj else None
                    log.append({k: e[k] for k in ('command', 'ref', 'verb', 'cands', 'gold', 'ambiguous', 'context_resolved')}
                               | {'t': round(now, 1), 'acted': obj, 'asked': asked, 'ok': obj in ok, 'wait': round(wait, 2),
                                  'latency_ms': [round(x) for x in lat]})
                    pending = m if m in MOVE_TO_HT else None
                    free_at = now + wait + (0 if pending else args.gap)
                    if pending and wait == 0:
                        start(AI, pending, step)
                        pending = None
                else:
                    free_at = now + 1.0
        actions = {}
        for idx in (AI, HUMAN):
            move = (0, 0)
            if tasks[idx] is not None:
                state, move, _ = tasks[idx](env_state(env, idx))
                # executors can wait forever (e.g. for a free cutboard), so cap each macro
                if state != HighTask.Working or step - started[idx] >= MACRO_TIMEOUT_STEPS:
                    tasks[idx] = None
                    if idx == AI:
                        free_at = max(free_at, now + args.gap)
            actions[env.sim_agents[idx].name] = move
        _, _, done, _ = env.step(actions, passed_time=STEP_SECONDS)
        if done:
            break
    sched = env.order_scheduler  # env.step always returns reward 0 here, so count orders from the scheduler
    return {'map': game_map, 'game': g, 'policy': args.name or policy, 'served': sched.successful_orders,
            'expired': sched.failed_orders, 'score': sched.reward, 'commands': log}


def summary(games):
    cmds = [c for gm in games for c in gm['commands']]
    amb = [c for c in cmds if c['ambiguous']]
    clear = [c for c in cmds if not c['ambiguous']]
    lat = sorted(x for c in cmds for x in c['latency_ms'])
    mean = lambda xs, f: sum(map(f, xs)) / len(xs) if xs else float('nan')
    pct = lambda q: lat[int(q * (len(lat) - 1))] if lat else 0
    return {'served': mean(games, lambda x: x['served']), 'expired': mean(games, lambda x: x['expired']),
            'commands': mean(games, lambda x: len(x['commands'])), 'ambiguous': mean(cmds, lambda c: c['ambiguous']),
            'right step': mean(cmds, lambda c: c['ok']), 'asks on ambiguous': mean(amb, lambda c: c['asked']),
            'asks on clear': mean(clear, lambda c: c['asked']), 'wait s': mean(cmds, lambda c: c['wait']),
            'latency p50 / p95 ms': f'{pct(0.5):.0f} / {pct(0.95):.0f}'}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--policy', required=True, choices=['idle', 'mind_reader', *RULES, 'model'])
    a.add_argument('--name', default=None, help='label for the results (default: the policy)')
    a.add_argument('--url', default=None, help='model server, e.g. http://localhost:8080/v1')
    a.add_argument('--model', default='default_model', help='model name the server expects')
    a.add_argument('--maps', default=','.join(LIVE_MAPS))
    a.add_argument('--games', type=int, default=10, help='games per map')
    a.add_argument('--first', type=int, default=FIRST_GAME, help='seed of the first game on each map')
    a.add_argument('--max-time', type=int, default=160, help='seconds per game (400 steps)')
    a.add_argument('--reply-delay', type=float, default=2.0, help='seconds the player needs to answer a question')
    a.add_argument('--gap', type=float, default=1.0, help='seconds between the AI finishing and the next command')
    a.add_argument('--latency-scale', type=float, default=1.0, help='0: the AI does not wait for its model (isolates the cost of latency)')
    a.add_argument('--wording', default='templates', choices=['templates', 'keyword-free'],
                   help='keyword-free: the player uses the phrasings of the keyword-free test set')
    a.add_argument('--out', default=None)
    args = a.parse_args()
    if args.wording == 'keyword-free':
        K.TEMPLATES = NEW  # command() and make() read the module's templates
    gen = API(args.model, workers=1, url=args.url) if args.policy == 'model' else None
    games = []
    for game_map in args.maps.split(','):
        for g in range(args.first, args.first + args.games):
            games.append(play(game_map, g, args.policy, gen, args))
            print(f"{game_map} game {g}: served {games[-1]['served']}, expired {games[-1]['expired']}, "
                  f"{len(games[-1]['commands'])} commands", flush=True)
    rows = {'all': summary(games)} | {m: summary([x for x in games if x['map'] == m]) for m in args.maps.split(',')}
    print(f"\n{args.name or args.policy}\n| maps | " + ' | '.join(rows['all']) + ' |\n|---|' + '---|' * len(rows['all']))
    for k, r in rows.items():
        print(f'| {k} | ' + ' | '.join(v if isinstance(v, str) else f'{v:.2f}' for v in r.values()) + ' |')
    if args.out:
        with open(args.out, 'w') as f:
            for x in games:
                f.write(json.dumps(x) + '\n')


if __name__ == '__main__':
    main()
