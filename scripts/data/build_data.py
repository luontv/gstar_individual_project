import argparse
import json
import os
import random

from askoract import clarify as K, prompt
from askoract.data import TEST_MAP, VAL_GAME, execute, row, split_of, turn1
from askoract.prompt import messages


CHAT = ['nice job!', 'good work', 'thanks', 'lol', "let's go!", 'we got this', 'oops my bad', 'what are the orders?',
        'how much time is left?']


# ask with the true candidates, simulate a reply, target the action the reply points to
def turn2(e, s, rng):
    q = {'options': e['cands']}
    text, val, ok = K.reply(q, e['gold'], e['cands'], e['verb'], rng)
    left = e['cands'] if val in (None, 'ANY') else K.remaining(q, val, e['cands'], e['verb'])
    final = left[0] if len(left) == 1 else K.best(e['verb'], left, s)  # silent or "either": most urgent
    return K.say(e['cands'], e['verb']), text, execute(K.macro(e['verb'], final)), ok


def refuse_example(s, rng):
    verb = rng.choice(K.VERBS)
    all_objs = list(K.RECIPE) if verb != 'Chop' else list(K.INGREDIENTS)
    missing = [o for o in all_objs if o not in K.objects(verb, s)]
    if not missing:
        return None
    obj = rng.choice(missing)
    cmd = rng.choice(K.TEMPLATES[(verb, 'explicit')]).format(x=obj.lower())
    return cmd, {'mode': 'refuse', 'reason': f'{K.macro(verb, obj)} is not possible right now.'}


def reply_example(s, rng):
    cmd = rng.choice(CHAT)
    if 'orders' in cmd:
        text = 'Orders: ' + (', '.join(o['name'].replace('Plated ', '') for o in s['orders']) or 'none') + '.'
    elif 'time' in cmd:
        text = f"About {max(0, int(200 - s['time']))} seconds left."
    else:
        text = 'On it!' if rng.random() < 0.5 else 'Thanks!'
    return cmd, {'mode': 'reply', 'text': text}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--states', default='data/states.jsonl')
    p.add_argument('--out', default='data/sft')
    p.add_argument('--n-train', type=int, default=5000, help='first-turn training examples')
    p.add_argument('--n-turn2', type=int, default=1500)
    p.add_argument('--n-other', type=int, default=300, help='refuse and reply examples each')
    p.add_argument('--n-test', type=int, default=2000, help='cap per test split')
    p.add_argument('--para', default=None, help='rewordings from paraphrase.py, applied to train only')
    p.add_argument('--p-para', type=float, default=0.7, help='share of training commands reworded')
    p.add_argument('--perturb', type=int, default=0, help='copies of each training moment with random orders')
    p.add_argument('--drop', type=int, default=0, help='copies of each training moment with some possible actions removed')
    p.add_argument('--coords', action='store_true', help='old prompt with grid coordinates (runs 1-2)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    prompt.COORDS = a.coords
    rng = random.Random(a.seed)
    states = [json.loads(l) for l in open(a.states)]
    # extra copies of training moments with random orders, so which options serve the orders isn't tied to the map.
    # they keep their game id, so they stay in train
    prng = random.Random(a.seed + 1)  # separate stream so the test sampling below doesn't change
    n_orig = len(states)
    for s in [s for s in states if s['map'] != TEST_MAP and int(s['game'].split('-')[1]) != VAL_GAME]:
        for _ in range(a.perturb):
            orders = [{'name': f'Plated {prng.choice(list(K.RECIPE))} Soup', 'rate': prng.uniform(0.05, 1.0)}
                      for _ in range(prng.randint(1, 4))]
            states.append(s | {'orders': orders, 'perturbed': True})
    # copies with some actions removed (as if out of reach), to teach "only one option -> just do it".
    # the prompt doesn't list ingredient tiles, so a missing "Chop X" still looks consistent
    for s in [s for s in states[:n_orig] if s['map'] != TEST_MAP and int(s['game'].split('-')[1]) != VAL_GAME]:
        for _ in range(a.drop):
            feasible = list(s['feasible'])
            for verb in K.VERBS:
                objs = [m for m in feasible if K.split(m)[0] == verb]
                if len(objs) > 1 and prng.random() < 0.6:
                    keep = set(prng.sample(objs, prng.randint(1, len(objs) - 1)))
                    feasible = [m for m in feasible if m in keep or K.split(m)[0] != verb]
            if feasible != s['feasible']:
                states.append(s | {'feasible': feasible, 'perturbed': True})
    out = {k: [] for k in ('train', 'val', 'test_phrasing', 'test_map')}

    def meta(e, kind):
        return {k: e[k] for k in ('state_id', 'command', 'verb', 'ref', 'family', 'cands', 'gold', 'gold_macro', 'ambiguous',
                                  'context_resolved', 'gold_is_best', 'value_conflict')} | {'kind': kind}

    # train / val: one intent per command; tests: every intent
    for all_golds, wanted in ((False, ('train', 'val')), (True, ('test_phrasing', 'test_map'))):
        for e in K.episodes(states, random.Random(a.seed), all_golds=all_golds):
            s = states[e['state_id']]
            sp = split_of(e, s)
            if sp in wanted:
                out[sp].append(row(messages(s, e['command']), turn1(e), meta(e, 'turn1')))

    # reword training commands (splits are already fixed above)
    if a.para:
        para = json.load(open(a.para))
        for r in out['train']:
            m = r['meta']
            forms = para.get(m['family'], [])
            if forms and rng.random() < a.p_para:
                key = (None, 'again') if m['ref'] == 'again' else (m['verb'], m['ref'])
                m['command'] = rng.choice(forms).format(x=m['gold'].lower()) if '{x}' in K.TEMPLATES[key][0] \
                    else rng.choice(forms)
                r['messages'] = messages(states[m['state_id']], m['command']) + r['messages'][-1:]

    # balance first-turn training data: about 45% ambiguous
    tr = out['train']
    amb = [r for r in tr if r['meta']['ambiguous']]
    clear = [r for r in tr if not r['meta']['ambiguous']]
    rng.shuffle(amb)
    rng.shuffle(clear)
    k = min(len(amb), int(a.n_train * 0.45))
    first = amb[:k] + clear[:a.n_train - k]

    # second turns, refusals, chit-chat
    extra = []
    for r in amb[:a.n_turn2]:
        e, s = r['meta'], states[r['meta']['state_id']]
        asked, replied, target, ok = turn2(e, s, rng)
        extra.append(row(messages(s, e['command'], asked, replied),
                         target, e | {'kind': 'turn2', 'ok': ok}))
    train_states = sorted({r['meta']['state_id'] for r in tr})
    for maker, kind in ((refuse_example, 'refuse'), (reply_example, 'reply')):
        n = 0
        while n < a.n_other:
            sid = rng.choice(train_states)
            ex = maker(states[sid], rng)
            if ex:
                extra.append(row(messages(states[sid], ex[0]), ex[1], {'state_id': sid, 'kind': kind}))
                n += 1
    out['train'] = first + extra
    rng.shuffle(out['train'])
    for sp in ('val', 'test_phrasing', 'test_map'):
        n = a.n_test if sp != 'val' else 600
        rows = out[sp]
        rng.shuffle(rows)
        amb = [r for r in rows if r['meta']['ambiguous']][:n // 2]  # at most half ambiguous
        out[sp] = amb + [r for r in rows if not r['meta']['ambiguous']][:n - len(amb)]
        rng.shuffle(out[sp])

    os.makedirs(a.out, exist_ok=True)
    for sp, rows in out.items():
        with open(os.path.join(a.out, f'{sp}.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        kinds = {}
        for r in rows:
            key = r['meta']['kind'] + ('/amb' if r['meta'].get('ambiguous') else '')
            kinds[key] = kinds.get(key, 0) + 1
        print(f'{sp:14s} {len(rows):5d}  {dict(sorted(kinds.items()))}')


if __name__ == '__main__':
    main()
