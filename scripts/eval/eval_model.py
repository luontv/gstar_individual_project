import argparse
import json
import random
import statistics
import time

from askoract import clarify as K, prompt
from askoract.backends import API, HF
from askoract.prompt import acted_obj, messages, parse, question
from askoract.reward import summarize, table, turn1_reward


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--split', default='data/test/test_phrasing_nc.jsonl')
    a.add_argument('--states', default='data/states_v2.jsonl')
    a.add_argument('--model', default='Qwen/Qwen3-1.7B')
    a.add_argument('--adapter', default=None)
    a.add_argument('--api', default=None, help='LiteLLM model name instead of a local model')
    a.add_argument('--url', default=None, help='with --api: local OpenAI-compatible server (e.g. http://localhost:8080/v1)')
    a.add_argument('--limit', type=int, default=0)
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--workers', type=int, default=16, help='parallel API calls')
    a.add_argument('--save', default=None, help='write raw outputs here (jsonl)')
    a.add_argument('--coords', action='store_true', help='second-turn prompts with grid coordinates (runs 1-2)')
    args = a.parse_args()
    prompt.COORDS = args.coords

    states = [json.loads(l) for l in open(args.states)]
    rows = [r for r in map(json.loads, open(args.split)) if r['meta']['kind'] == 'turn1']
    rows = rows[:args.limit] if args.limit else rows
    gen = API(args.api, workers=args.workers, url=args.url) if args.api else HF(args.model, args.adapter, args.batch)
    name = args.api or (args.adapter or args.model)

    t = time.time()
    outs1 = gen([r['messages'][:-1] for r in rows])
    rng = random.Random(0)
    recs, follow, log = [None] * len(rows), [], []
    for i, (r, o) in enumerate(zip(rows, outs1)):
        e, out = r['meta'], parse(o)
        log.append({'i': i, 'turn1': o})
        if out.get('mode') == 'ask':
            q = question(out)
            text, _, ok = K.reply(q, e['gold'], e['cands'], e['verb'], rng)
            ig = K.info_gain(q, e['cands'], e['verb'])
            asked = out.get('question') or K.say(q['options'] or ['it'], e['verb'])
            follow.append((i, messages(states[e['state_id']], e['command'], asked, text), ok, ig))
            log[-1]['reply'] = text
        else:
            final = acted_obj(out, e['verb'])
            recs[i] = {'e': e, 'asked': False, 'ig': 0.0, 'ok': final == e['gold'],
                       'reward': turn1_reward(e, False, 0.0, final)}
    outs2 = gen([f[1] for f in follow])
    for (i, _, ok, ig), o in zip(follow, outs2):
        e = rows[i]['meta']
        final = acted_obj(parse(o), e['verb'])
        recs[i] = {'e': e, 'asked': True, 'ig': ig, 'ok': final in ok, 'reward': turn1_reward(e, True, ig, final)}
        log[i]['turn2'] = o
    elapsed = time.time() - t

    bad = sum(not parse(o).get('mode') for o in outs1)
    print(f'{name}: {len(rows)} commands, {len(follow)} questions, {bad} unparseable, {elapsed / 60:.1f} min')
    print(table([summarize(name, recs)]))

    if isinstance(gen, HF):  # one request at a time, like in a real game
        sample = random.Random(1).sample(rows, min(30, len(rows)))
        ms = []
        for r in sample:
            t0 = time.time()
            gen([r['messages'][:-1]], batch=1)
            ms.append(1000 * (time.time() - t0))
        ms.sort()
        print(f'latency per command: p50 {statistics.median(ms):.0f} ms, p95 {ms[int(0.95 * (len(ms) - 1))]:.0f} ms')
    if args.save:
        with open(args.save, 'w') as f:
            for row in log:
                f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
