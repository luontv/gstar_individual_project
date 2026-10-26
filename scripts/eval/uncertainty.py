import json
import os
import random
import re
import statistics

from askoract import clarify as K
from askoract.policies import AGAIN, never_ask_best, oracle_rule, state_rule, text_rule
from askoract.prompt import acted_obj, parse
from askoract.reward import reply_reward, turn1_reward

SPLITS = ('test_phrasing', 'test_map', 'test_free', 'test_new')
MODELS = [('luna', 'gpt-5.6-luna'), ('base', 'Qwen3-1.7B zero-shot'), ('base06', 'Qwen3-0.6B zero-shot'),
          ('v3', 'SFT-1.7B'), ('sft06', 'SFT-0.6B seed 42'), ('sft06s1', 'SFT-0.6B seed 1'), ('sft06s2', 'SFT-0.6B seed 2')]
RULES = [(never_ask_best, 'Never ask'), (text_rule, 'Keyword rule'), (state_rule, 'Keyword + state rule'), (oracle_rule, 'Oracle rule')]
PAIRS = [('SFT-1.7B', 'Keyword rule'), ('SFT-1.7B', 'gpt-5.6-luna'), ('SFT-0.6B seed 42', 'Keyword rule'),
         ('SFT-1.7B', 'SFT-0.6B seed 42'), ('SFT-1.7B', 'Keyword + state rule'), ('SFT-1.7B', 'Keyword + state rule, own verb')]
LIVE = [('never_ask', 'Never ask'), ('keyword', 'Keyword rule'), ('perfect_rule', 'Oracle rule'),
        ('mind_reader', 'Intent oracle'), ('base-mlx8', 'Qwen3-1.7B zero-shot'), ('base06-mlx8', 'Qwen3-0.6B zero-shot'),
        ('sft-v3-mlx8', 'SFT-1.7B'), ('sft-v3-mlx8-nolatency', 'SFT-1.7B, no latency'), ('sft06-mlx8', 'SFT-0.6B')]
# the same live games with the keyword-free phrasings (live.py --wording keyword-free)
LIVE_KF = [('kf_never_ask', 'Never ask'), ('kf_keyword', 'Keyword rule'), ('kf_perfect_rule', 'Oracle rule'),
           ('kf_base-mlx8', 'Qwen3-1.7B zero-shot'), ('kf_sft-v3-mlx8', 'SFT-1.7B'), ('kf_sft06-mlx8', 'SFT-0.6B')]
RL = [('grpo', 'sftmore'), ('grpo', 'sftall'), ('grpo', 'sftlow'), ('grpo06', 'sftall06'), ('sft06g', 'sft06')]
GAME_MIN = 160 / 60
# verb words from the training commands only
VERB_WORDS = [('Plate', r'\bplate'), ('Serve', r'\b(serve|deliver|send)'), ('Cook', r'\b(cook|start)'),
              ('Prepare', r'\b(prepare|ingredients)'), ('Chop', r'\b(chop|veggie)')]


# first decision of a scripted policy on each row
def rule_decisions(policy, rows, states):
    rng = random.Random(0)
    return [policy(e, states[e['state_id']], rng) for e in rows]


# saved first replies of a model
def model_outputs(prefix, split):
    return [json.loads(l)['turn1'] for l in open(f'results/{prefix}_{split}.jsonl')]


# turn-1 reward of each row; correct_guess replaces -1 for an act on an ambiguous command that hits the intended object
def rule_rewards(decs, rows, correct_guess=-1.0):
    out = []
    for (kind, x), e in zip(decs, rows):
        if kind == 'ask':
            out.append(turn1_reward(e, True, K.info_gain({'options': x}, e['cands'], e['verb']), None))
        else:
            out.append(correct_guess if e['ambiguous'] and x == e['gold'] else turn1_reward(e, False, 0.0, x))
    return out


def model_rewards(outs, rows, correct_guess=-1.0):
    r = [reply_reward(o, e)[0] for o, e in zip(outs, rows)]
    if correct_guess != -1.0:
        r = [correct_guess if e['ambiguous'] and parse(o).get('mode') != 'ask' and acted_obj(parse(o), e['verb']) == e['gold']
             else x for x, o, e in zip(r, outs, rows)]
    return r


# first reply is right: asks when the command is ambiguous, otherwise takes the intended step
def rule_right(decs, rows):
    return [kind == 'ask' if e['ambiguous'] else kind == 'act' and x == e['gold'] for (kind, x), e in zip(decs, rows)]


def model_right(outs, rows):
    return [parse(o).get('mode') == 'ask' if e['ambiguous'] else reply_reward(o, e)[0] == 1 for o, e in zip(outs, rows)]


# keyword + state rule that also takes the verb from keywords instead of the label; a wrong or missing verb is a wrong step
def own_verb_rewards(rows, states):
    out = []
    for e in rows:
        s = states[e['state_id']]
        v = next((v for v, w in VERB_WORDS if re.search(w, e['command'])), None)
        if AGAIN.search(e['command']) and s['ai_hist']:
            v = K.split(s['ai_hist'][-1])[0]
        if v is None or not K.objects(v, s):
            out.append(turn1_reward(e, False, 0.0, None))
            continue
        kind, x = state_rule(e | {'verb': v}, s, None)
        ok = v == e['verb']
        if kind == 'ask':
            out.append(turn1_reward(e, True, K.info_gain({'options': x}, e['cands'], e['verb']) if ok else 0.0, None))
        else:
            out.append(turn1_reward(e, False, 0.0, x if ok else None))
    return out


# 95% interval of a mean, resampling whole game moments (rows of one moment are not independent)
def boot(rows, vals, n=2000):
    groups = {}
    for e, v in zip(rows, vals):
        groups.setdefault(e['state_id'], []).append(v)
    g, rng, means = list(groups.values()), random.Random(0), []
    for _ in range(n):
        pick = [g[rng.randrange(len(g))] for _ in g]
        means.append(sum(map(sum, pick)) / sum(map(len, pick)))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def offline(states):
    for sp in SPLITS:
        rows = [r['meta'] for r in map(json.loads, open(f'data/test/{sp}_nc.jsonl')) if r['meta']['kind'] == 'turn1']
        amb = [e for e in rows if e['ambiguous']]
        verbs = {v: sum(e['verb'] == v for e in amb) for v in sorted({e['verb'] for e in amb})}
        fams = {e.get('family') for e in amb}
        print(f'\n## {sp}: {len(rows)} rows, {len({e["state_id"] for e in rows})} game moments; ambiguous {len(amb)} rows from '
              f'{len(fams)} template families, {len({e["command"] for e in amb})} distinct commands; verbs {verbs}; '
              f'ambiguous rows whose candidates are equally urgent: {sum(K.value(e["verb"], e["cands"][0], states[e["state_id"]]) == K.value(e["verb"], e["cands"][-1], states[e["state_id"]]) for e in amb)}')
        decs = {name: rule_decisions(p, rows, states) for p, name in RULES}
        outs = {name: model_outputs(pre, sp) for pre, name in MODELS if os.path.exists(f'results/{pre}_{sp}.jsonl')}
        vals = {name: rule_rewards(d, rows) for name, d in decs.items()}
        vals['Keyword + state rule, own verb'] = own_verb_rewards(rows, states)
        vals.update({name: model_rewards(o, rows) for name, o in outs.items()})
        print('| policy | turn-1 reward | 95% CI | correct guess scored +1 | right decision |')
        print('|---|---|---|---|---|')
        for name, d in decs.items():
            lo, hi = boot(rows, vals[name])
            alt = statistics.mean(rule_rewards(d, rows, 1.0))
            print(f'| {name} | {statistics.mean(vals[name]):.2f} | [{lo:.2f}, {hi:.2f}] | {alt:.2f} | {statistics.mean(rule_right(d, rows)):.1%} |')
        lo, hi = boot(rows, vals['Keyword + state rule, own verb'])
        print(f"| Keyword + state rule, own verb | {statistics.mean(vals['Keyword + state rule, own verb']):.2f} | [{lo:.2f}, {hi:.2f}] | - | - |")
        for name, o in outs.items():
            lo, hi = boot(rows, vals[name])
            alt = statistics.mean(model_rewards(o, rows, 1.0))
            print(f'| {name} | {statistics.mean(vals[name]):.2f} | [{lo:.2f}, {hi:.2f}] | {alt:.2f} | {statistics.mean(model_right(o, rows)):.1%} |')
        for a, b in (p for p in PAIRS if p[0] in vals and p[1] in vals):
            d = [x - y for x, y in zip(vals[a], vals[b])]
            lo, hi = boot(rows, d)
            print(f'paired difference {a} - {b}: {statistics.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]')


# GRPO against its same-data SFT controls, and two runs of one configuration (seed 42)
def rl():
    print('\n## GRPO vs controls, paired turn-1 reward difference (95% CI)')
    for sp in ('val',) + SPLITS[:3]:
        rows = [r['meta'] for r in map(json.loads, open(f'data/test/{sp}_nc.jsonl')) if r['meta']['kind'] == 'turn1']
        for a, b in RL:
            d = [x - y for x, y in zip(model_rewards(model_outputs(a, sp), rows), model_rewards(model_outputs(b, sp), rows))]
            lo, hi = boot(rows, d)
            print(f'{sp:14s} {a} - {b}: {statistics.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]')


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


def live(runs=LIVE, title='Live games'):
    games = {k: {(g['map'], g['game']): g for g in map(json.loads, open(f'results/live_{k}.jsonl'))} for k, _ in runs}
    never, keyword = runs[0][0], runs[1][0]
    print(f'\n## {title}')
    print('| policy | orders ± SE (games) | vs never ask (95% CI) | vs keyword (95% CI) | step acc. | step acc., ambiguous | '
          'questions / min | call latency p50 / p95 | command latency p50 / p95 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for k, name in runs:
        g = games[k]
        orders = [x['served'] for x in g.values()]
        cmds = [c for x in g.values() for c in x['commands']]
        amb = [c for c in cmds if c['ambiguous']]
        row = [name, f'{statistics.mean(orders):.2f} ± {statistics.stdev(orders) / len(orders) ** 0.5:.2f} (n={len(orders)})']
        for ref in (never, keyword):
            d = [g[s]['served'] - games[ref][s]['served'] for s in g if s in games[ref]]
            half = 1.96 * statistics.stdev(d) / len(d) ** 0.5
            row.append('-' if k == ref else f'{statistics.mean(d):+.2f} [{statistics.mean(d) - half:+.2f}, {statistics.mean(d) + half:+.2f}]')
        row += [f'{statistics.mean(c["ok"] for c in cmds):.3f}', f'{statistics.mean(c["ok"] for c in amb):.2f} (n={len(amb)})',
                f'{sum(c["asked"] for c in cmds) / (len(g) * GAME_MIN):.2f}']
        calls = [ms for c in cmds for ms in c.get('latency_ms') or []]
        per_cmd = [sum(c['latency_ms']) for c in cmds if c.get('latency_ms')]
        row += [f'{pct(calls, 0.5):.0f} / {pct(calls, 0.95):.0f}' if calls else '-',
                f'{pct(per_cmd, 0.5):.0f} / {pct(per_cmd, 0.95):.0f}' if per_cmd else '-']
        print('| ' + ' | '.join(row) + ' |')
        acc = lambda x: statistics.mean(c['ok'] for c in x['commands'])
        for ref in {never: [], keyword: [never]}.get(k, [never, keyword]):
            d = [acc(g[s]) - acc(games[ref][s]) for s in g if s in games[ref]]
            half = 1.96 * statistics.stdev(d) / len(d) ** 0.5
            print(f'  {name}: step accuracy vs {dict(runs)[ref].lower()} {statistics.mean(d):+.3f} [{statistics.mean(d) - half:+.3f}, '
                  f'{statistics.mean(d) + half:+.3f}] over {len(d)} shared games')
        first = [(c['asked'], c['latency_ms'][0]) for c in cmds if c.get('latency_ms')]
        if first and any(a for a, _ in first):
            ask_ms = statistics.median(ms for a, ms in first if a)
            act_ms = statistics.median(ms for a, ms in first if not a)
            print(f'  {name}: first call median {ask_ms:.0f} ms when asking, {act_ms:.0f} ms when acting')


if __name__ == '__main__':
    offline([json.loads(l) for l in open('data/states_v2.jsonl')])
    rl()
    live()
    live(LIVE_KF, 'Live games, keyword-free wording')
