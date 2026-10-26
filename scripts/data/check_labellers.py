import argparse
import json
import random

from askoract import clarify as K
from askoract.backends import API
from askoract.labelling import LABEL, kappa, macros
from askoract.prompt import observe


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--split', default='data/test/test_phrasing.jsonl')
    a.add_argument('--states', default='data/states.jsonl')
    a.add_argument('--models', default='qwen/qwen3.8-flash,deepseek-v4-flash')
    a.add_argument('--n', type=int, default=300)
    a.add_argument('--out', default='results/labeller_check.jsonl')
    args = a.parse_args()
    states = [json.loads(l) for l in open(args.states)]
    rows = [r['meta'] for r in map(json.loads, open(args.split)) if r['meta']['kind'] == 'turn1']
    seen, sample = set(), []
    for m in random.Random(0).sample(rows, len(rows)):  # one row per command (tests repeat ambiguous ones per intent)
        key = (m['state_id'], m['command'])
        if key not in seen and len(sample) < args.n:
            seen.add(key)
            sample.append(m)
    msgs = [[{'role': 'user', 'content': LABEL.format(situation=observe(states[m['state_id']], coords=True),
                                                      message=m['command'])}] for m in sample]
    out = {}
    for model in args.models.split(','):
        res = API(model, workers=24)(msgs)
        sets = [macros(o, states[m['state_id']]['feasible']) for o, m in zip(res, sample)]
        ok = [(m, [K.split(x)[1] for x in st if K.split(x)[0] == m['verb']]) for m, st in zip(sample, sets) if st is not None]
        pairs = [(m['ambiguous'], len(objs) > 1) for m, objs in ok]
        out[model] = sets
        print(f"{model:32s} answered {len(ok)}/{len(sample)}  clear/ambiguous acc {sum(x == y for x, y in pairs) / len(pairs):.2f}"
              f"  kappa {kappa(pairs):.2f}  exact set {sum(set(o) == set(m['cands']) for m, o in ok) / len(ok):.2f}"
              f"  intended in set {sum(m['gold'] in o for m, o in ok) / len(ok):.2f}"
              f"  says ambiguous {sum(y for _, y in pairs) / len(pairs):.2f} (truth {sum(x for x, _ in pairs) / len(pairs):.2f})", flush=True)
    with open(args.out, 'w') as f:
        for i, m in enumerate(sample):
            f.write(json.dumps({'meta': m, **{k: v[i] for k, v in out.items()}}) + '\n')


if __name__ == '__main__':
    main()
