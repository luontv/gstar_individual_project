import argparse
import json
import random

from askoract import clarify as K
from askoract.policies import POLICIES
from askoract.reward import summarize, table, turn1_reward


# play every command with a scripted policy and the simulated player
def run(policy, eps, states, seed=0):
    rng = random.Random(seed)
    recs = []
    for e in eps:
        s = states[e['state_id']]
        kind, x = policy(e, s, rng)
        ok_set, ig = [e['gold']], 0.0
        if kind == 'ask':
            q = {'options': x}
            _, val, ok_set = K.reply(q, e['gold'], e['cands'], e['verb'], rng)
            final = K.settle(q, val, e['cands'], e['verb'], s)
            ig = K.info_gain(q, e['cands'], e['verb'])
        else:
            final = x
        recs.append({'e': e, 'asked': kind == 'ask', 'ig': ig, 'ok': final in ok_set,
                     'reward': turn1_reward(e, kind == 'ask', ig, final)})
    return summarize(policy.__name__, recs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--states', default='data/states.jsonl')
    p.add_argument('--split', default=None, help='score on the first-turn rows of a split file instead of all commands')
    a = p.parse_args()
    states = [json.loads(l) for l in open(a.states)]
    if a.split:
        eps = [r['meta'] for r in map(json.loads, open(a.split)) if r['meta']['kind'] == 'turn1']
    else:
        eps = list(K.episodes(states, random.Random(0), all_golds=True))
    print(f"{len(eps)} commands, {sum(e['ambiguous'] for e in eps)} ambiguous, "
          f"{sum(e['context_resolved'] for e in eps)} clear only from context, "
          f"{sum(e['ambiguous'] and e['value_conflict'] for e in eps)} ambiguous where the intended option is not the most urgent\n")
    print(table([run(pol, eps, states) for pol in POLICIES]))
