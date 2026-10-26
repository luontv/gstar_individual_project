import argparse
import json
import os
import random
import time

from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from askoract.prompt import parse, prompt_text
from askoract.reward import reply_reward


def decision(text):
    out = parse(text)
    return out.get('mode'), out.get('macro') or tuple(sorted(map(str, out.get('options') or [])))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', required=True)
    a.add_argument('--split', default='data/test/test_map_nc.jsonl')
    a.add_argument('--hf', default=None, help='saved eval_model.py outputs on the same split')
    a.add_argument('--n', type=int, default=200)
    args = a.parse_args()
    model, tok = load(args.model)
    rows = [json.loads(l) for l in open(args.split)]
    hf = [json.loads(l)['turn1'] for l in open(args.hf)] if args.hf else None
    idx = random.Random(0).sample(range(len(rows)), min(args.n, len(rows)))
    ids = [tok.encode(prompt_text(tok, rows[i]['messages'][:-1])) for i in idx]
    n_pre = len(os.path.commonprefix(ids)) - 1  # leave at least one new token per command

    def timed(fn):
        for k in range(3):  # warm-up
            fn(k)
        ms, outs = [], []
        for k in range(len(ids)):
            t = time.perf_counter()
            outs.append(fn(k))
            ms.append(1000 * (time.perf_counter() - t))
        return ms, outs

    full_ms, outs = timed(lambda k: generate(model, tok, ids[k], max_tokens=80))
    cache = make_prompt_cache(model)
    generate(model, tok, ids[0][:n_pre], max_tokens=1, prompt_cache=cache)
    trim_prompt_cache(cache, cache[0].offset - n_pre)

    def cached(k):
        text = generate(model, tok, ids[k][n_pre:], max_tokens=80, prompt_cache=cache)
        trim_prompt_cache(cache, cache[0].offset - n_pre)
        return text

    cached_ms, outs_c = timed(cached)
    p = lambda xs, q: sorted(xs)[int(q * (len(xs) - 1))]
    print(f'{args.model} on {args.split}: {len(ids)} commands, prompt {sum(map(len, ids)) / len(ids):.0f} tokens '
          f'({n_pre} shared), reply {sum(len(tok.encode(o)) for o in outs) / len(outs):.0f} tokens')
    for name, ms in (('whole prompt', full_ms), ('shared start cached', cached_ms)):
        print(f'  {name:20s} p50 {p(ms, 0.5):.0f} ms  p95 {p(ms, 0.95):.0f} ms  max {max(ms):.0f} ms', end='')
        for mode in ('execute', 'ask'):
            sub = [m for m, o in zip(ms, outs) if decision(o)[0] == mode]
            if sub:
                print(f' | {mode} ({len(sub)}): p50 {p(sub, 0.5):.0f}, p95 {p(sub, 0.95):.0f}', end='')
        print()
    print(f'  cached vs whole prompt: same decision {sum(decision(x) == decision(y) for x, y in zip(outs, outs_c)) / len(outs):.2f}')
    r = lambda texts: sum(reply_reward(t, rows[i]['meta'])[0] for t, i in zip(texts, idx)) / len(idx)
    print(f'  turn-1 reward: MLX {r(outs):.3f}', end='')
    if hf:
        ref = [hf[i] for i in idx]
        print(f', Hugging Face {r(ref):.3f}; same decision as Hugging Face '
              f'{sum(decision(x) == decision(y) for x, y in zip(outs, ref)) / len(outs):.2f}', end='')
    print()


if __name__ == '__main__':
    main()
