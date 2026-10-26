import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from askoract import clarify as K, prompt
from askoract.backends import API
from askoract.data import TEST_MAP, VAL_GAME, row, turn1
from askoract.labelling import LABEL, kappa, macros
from askoract.prompt import messages, observe

WRITER = 'kimi-k2.6'
LABELLERS = ('gemini-3.5-flash',)  # checked against exact labels: kappa 0.88

WRITE = """You are the pink chef in a fast cooperative cooking game. Your AI teammate (blue) acts on your chat messages.
Recipes: Alice = lettuce + onion; Bob = lettuce + tomato; Cathy = onion + tomato; David = lettuce + onion + tomato.

Situation:
{situation}

You want your teammate to do: {macro}
Write four chat messages you might send for this:
1. the way you would naturally type it mid-game;
2. a version that relies on the situation instead of naming things ("that one", "the other one", "this", "the urgent one", ...);
3. the shortest thing you might type in a hurry, even if your teammate might have to ask what you mean;
4. a version that leaves out which ingredient or soup you mean, the way people do when they assume the teammate knows
   ("chop something", "make the soup", "serve it", ...).
Answer with a JSON list of four strings only."""
STYLES = ('natural', 'context', 'minimal', 'underspecified')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--states', default='data/states.jsonl')
    a.add_argument('--n', type=int, default=300, help='commands to write before filtering')
    a.add_argument('--out', default='data/sft/test_free.jsonl')
    a.add_argument('--raw', default='data/freeform_raw.jsonl')
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--relabel', action='store_true', help='reuse the messages in --raw; only label them again')
    a.add_argument('--labellers', default=','.join(LABELLERS), help='one or two labellers')
    args = a.parse_args()
    labellers = args.labellers.split(',')
    rng = random.Random(args.seed)
    states = [json.loads(l) for l in open(args.states)]
    test_ids = [i for i, s in enumerate(states)
                if s['map'] == TEST_MAP or int(s['game'].split('-')[1]) == VAL_GAME]

    if args.relabel:
        items = [(r['state_id'], r['verb'], r['gold'], r['style'], r['message']) for r in map(json.loads, open(args.raw))]
        picks = sorted({i[0] for i in items})
    else:
        # balanced over verbs; prefer moments where the verb has several options
        picks = []
        while len(picks) < args.n:
            sid = rng.choice(test_ids)
            s = states[sid]
            verb = K.VERBS[len(picks) % len(K.VERBS)]
            objs = K.objects(verb, s)
            if objs and (len(objs) > 1 or rng.random() < 0.3):
                picks.append((sid, verb, rng.choice(objs)))

        situation = lambda s: observe(s, coords=True).rsplit('\nPossible actions now', 1)[0]  # the player doesn't see the AI's action list
        write_msgs = [[{'role': 'user', 'content': WRITE.format(situation=situation(states[sid]), macro=K.macro(v, o))}]
                      for sid, v, o in picks]
        items = []  # (state_id, verb, gold, style, message)
        print(f'writing {len(write_msgs)} situations with {WRITER}...', flush=True)
        for (sid, v, o), out in zip(picks, API(WRITER, workers=24)(write_msgs)):
            try:
                versions = [str(x).strip().strip('"') for x in json.loads(re.search(r'\[.*\]', out, re.S).group(0))][:4]
            except (AttributeError, json.JSONDecodeError):
                continue
            items += [(sid, v, o, st, m) for st, m in zip(STYLES, versions) if m]
    label_msgs = [[{'role': 'user', 'content': LABEL.format(situation=observe(states[sid], coords=True), message=w)}]
                  for sid, _, _, _, w in items]
    print(f'labelling {len(label_msgs)} messages with {labellers}...', flush=True)
    with ThreadPoolExecutor(len(labellers)) as ex:
        labels = dict(zip(labellers, ex.map(lambda m: API(m, workers=24)(label_msgs), labellers)))

    raw, rows, pairs, styles, agree_all, n_valid = [], [], [], [], 0, 0
    for i, (sid, verb, gold, style, w) in enumerate(items):
        s = states[sid]
        sets = [macros(labels[m][i], s['feasible']) for m in labellers]
        raw.append({'state_id': sid, 'verb': verb, 'gold': gold, 'style': style, 'message': w,
                    **{m: st for m, st in zip(labellers, sets)}})
        if not w or any(st is None for st in sets):
            continue
        n_valid += 1
        objs = [[K.split(m)[1] for m in st if K.split(m)[0] == verb] for st in sets]
        amb = [len(o) > 1 for o in objs]
        agree_all += set(objs[0]) == set(objs[-1])
        pairs.append((amb[0], amb[-1]))
        styles.append(style)
        if amb[0] != amb[-1] or not all(gold in o for o in objs):
            continue
        cands = sorted(set().union(*objs), key=lambda o: K.objects(verb, s).index(o))
        for g in (cands if amb[0] else [gold]):
            vals = {c: K.value(verb, c, s) for c in cands}
            top = max(vals.values())
            e = {'state_id': sid, 'command': w, 'verb': verb, 'ref': 'free', 'family': f'free:{style}',
                 'cands': cands, 'gold': g, 'gold_macro': K.macro(verb, g), 'ambiguous': amb[0],
                 'context_resolved': not amb[0] and len(K.objects(verb, s)) > 1 and gold.lower() not in w.lower(),
                 'gold_is_best': vals[g] == top and list(vals.values()).count(top) == 1,
                 'value_conflict': vals[g] < top}
            rows.append((s, w, e))

    with open(args.raw, 'w') as f:
        for r in raw:
            f.write(json.dumps(r) + '\n')
    for path, coords in ((args.out, True), (args.out.replace('.jsonl', '_nc.jsonl'), False)):  # runs 1-2 / run 3+ prompts
        prompt.COORDS = coords
        with open(path, 'w') as f:
            for s, w, e in rows:
                f.write(json.dumps(row(messages(s, w), turn1(e), e | {'kind': 'turn1'})) + '\n')
    n_amb = sum(e['ambiguous'] for _, _, e in rows)
    po = sum(x == y for x, y in pairs) / len(pairs)
    p1, p2 = (sum(p[k] for p in pairs) / len(pairs) for k in (0, 1))
    by_style = {st: [p for p, x in zip(pairs, styles) if x == st] for st in STYLES}
    by_style = {st: f'{sum(a and b for a, b in v) / max(len(v), 1):.2f}' for st, v in by_style.items()}
    print(f'{len(items)} messages for {len(picks)} situations, {n_valid} labelled by all of {labellers}; clear/ambiguous agreement {po:.2f} (kappa {kappa(pairs):.2f}), '
          f'same set {agree_all / n_valid:.2f}; ambiguous per labeller {p1:.2f} / {p2:.2f}')
    print(f'labelled ambiguous, by style: {by_style}')
    print(f'kept {len(rows)} rows ({n_amb} ambiguous, one per intent) in {args.out}; raw in {args.raw}')


if __name__ == '__main__':
    main()
