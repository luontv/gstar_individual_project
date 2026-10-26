import json
import sys

from askoract import clarify as K
from askoract.prompt import acted_obj, parse, question


# candidates with the highest urgency
def top(e, s):
    v = {o: K.value(e['verb'], o, s) for o in e['cands']}
    return {o for o in v if v[o] == max(v.values())}


# turn-1 decisions on the rows a model answered (API errors dropped): asks on ambiguous and context-clear
# commands, question IG, and how often an act on an ambiguous command takes a most urgent candidate
def main():
    split, paths = sys.argv[1], sys.argv[2:]
    states = [json.loads(l) for l in open('data/states_v2.jsonl')]
    rows = [r['meta'] for r in map(json.loads, open(split)) if r['meta']['kind'] == 'turn1']
    print(split)
    for p in paths:
        d = [(e, parse(o)) for e, o in zip(rows, (json.loads(l)['turn1'] for l in open(p))) if not o.startswith('ERROR')]
        amb = [(e, x) for e, x in d if e['ambiguous']]
        ctx = [x for e, x in d if e['context_resolved']]
        asks = [(e, x) for e, x in amb if x.get('mode') == 'ask']
        ig = [K.info_gain(question(x), e['cands'], e['verb']) for e, x in asks]
        acts = [(e, states[e['state_id']], acted_obj(x, e['verb'])) for e, x in amb if x.get('mode') != 'ask']
        acts = [(e, top(e, s), obj) for e, s, obj in acts if len(top(e, s)) < len(e['cands'])]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        print(f'{p:45s} answered {len(d):4d}  asks: ambiguous {len(asks) / len(amb):.2f}  '
              f'context-clear {mean([x.get("mode") == "ask" for x in ctx]):.2f}  IG {mean(ig):.2f}  '
              f'acts on most urgent {mean([obj in t for e, t, obj in acts]):.2f} '
              f'(chance {mean([len(t) / len(e["cands"]) for e, t, obj in acts]):.2f}, n {len(acts)})')


if __name__ == '__main__':
    main()
