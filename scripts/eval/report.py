import argparse
import glob
import json
import os
import re

COLS = ('intent acc', 'acc | gold best', 'acc | gold not best', 'ask P', 'ask R', 'ask F1', 'over-ask',
        'over-ask | context', 'question IG', 'turn-1 reward')  # same order as reward.summarize
POLICIES = [  # (file prefix, row name in a multi-row file, label)
    ('rules', 'never_ask_best', 'Never ask, most urgent'), ('rules', 'always_ask', 'Always ask'),
    ('rules', 'text_rule', 'Keyword rule'), ('rules', 'state_rule', 'Keyword + state rule'), ('rules', 'oracle_rule', 'Perfect rule (upper bound)'),
    ('base', None, 'Qwen3-1.7B untrained'), ('luna', None, 'gpt-5.6-luna'), ('v3', None, 'SFT run 3'),
    ('sftmore', None, 'SFT run 3 + more SFT (control)'), ('grpo', None, 'SFT run 3 + GRPO'),
    ('sftall', None, 'SFT run 3 + full SFT recipe (fair control)'), ('sftlow', None, 'SFT run 3 + more SFT at lr 2e-5'),
    ('base06', None, 'Qwen3-0.6B untrained'), ('sft06', None, 'Qwen3-0.6B SFT (run-3 recipe)'),
    ('sft06s1', None, 'Qwen3-0.6B SFT, seed 1'), ('sft06s2', None, 'Qwen3-0.6B SFT, seed 2'),
    ('sft06g', None, 'Qwen3-0.6B SFT, seed 42 (GRPO start)'), ('grpo06', None, 'Qwen3-0.6B SFT + GRPO'),
    ('sftall06', None, 'Qwen3-0.6B SFT + full SFT recipe (fair control)')]
SPLITS = ('val', 'test_phrasing', 'test_map', 'test_free', 'test_new')
SHOW = {'intent acc': 'Intent acc', 'acc | gold not best': 'Acc, intended != most urgent', 'ask P': 'Ask P', 'ask R': 'Ask R',
        'ask F1': 'Ask F1', 'over-ask': 'Over-ask', 'over-ask | context': 'Over-ask, context-clear', 'turn-1 reward': 'Turn-1 reward'}
STEP = re.compile(r'step (\d+)/\d+\s+reward (\S+)\s+ask (\S+)\s+groups with signal (\d+)/(\d+)\s+trained (\d+)\s+len (\d+)\s+(\S+) min')


# row name -> metrics (and latency if printed) from the markdown table in a result file
def table_rows(path):
    out, latency = {}, None
    for line in open(path, errors='ignore'):
        m = re.search(r'p50 (\d+) ms, p95 (\d+) ms', line)
        if m:
            latency = f'{m.group(1)} / {m.group(2)}'
        parts = [p.strip() for p in line.strip().strip('|').split('|')]
        try:
            out[parts[0]] = dict(zip(COLS, map(float, parts[-len(COLS):])))
        except ValueError:
            continue
    for v in out.values():
        v['latency'] = latency
    return out


def results(folder):
    got = {}
    for prefix, name, label in POLICIES:
        for sp in SPLITS:
            path = os.path.join(folder, f'{prefix}_{sp}.txt')
            if os.path.exists(path):
                t = table_rows(path)
                if t:
                    got[label, sp] = t[name] if name else next(iter(t.values()))
    return got


def trend(log, window):
    steps = [tuple(map(float, m.groups())) for m in map(STEP.search, open(log, errors='ignore').read().replace('\r', '\n').splitlines()) if m]
    print(f'## GRPO training, {log} ({len(steps)} logged steps, means over {window}-step windows)\n')
    print('| Steps | Sampled reward | Ask rate | Groups with signal | Replies trained / step | Reply tokens |\n|---|---|---|---|---|---|')
    for k in sorted({int(st[0] - 1) // window for st in steps}):  # by step number: the log may hold only snapshots
        w = [st for st in steps if int(st[0] - 1) // window == k]
        mean = lambda col: sum(s[col] for s in w) / len(w)
        print(f'| {int(w[0][0])}-{int(w[-1][0])} | {mean(1):.3f} | {mean(2):.2f} | {mean(3) / w[0][4]:.2f} | {mean(5):.0f} | {mean(6):.0f} |')
    if steps:
        print(f'\nTraining time {steps[-1][7]:.0f} min to step {steps[-1][0]:.0f} ({60 * steps[-1][7] / steps[-1][0]:.0f} s/step).\n')


# orders served (with SE) and the paired difference to the base policy on the same seeds
def live(folder, base='keyword'):
    runs = {os.path.basename(f)[5:-6]: {(g['map'], g['game']): g for g in map(json.loads, open(f))}
            for f in sorted(glob.glob(os.path.join(folder, 'live_*.jsonl')))}
    if not runs:
        return
    se = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) ** 0.5
    print(f'## Live games\n\n| Policy | Games | Orders served (+- SE) | vs {base}, same seeds (+- SE) | Right step | '
          'Asks on ambiguous | Asks on clear | AI wait per command (s) | Model latency p50 / p95 ms |\n|---|' + '---|' * 8)
    for name, games in runs.items():
        served = [g['served'] for g in games.values()]
        cmds = [c for g in games.values() for c in g['commands']]
        rate = lambda xs, f: f'{sum(map(f, xs)) / len(xs):.2f}' if xs else '-'
        amb, clear = [c for c in cmds if c['ambiguous']], [c for c in cmds if not c['ambiguous']]
        lat = sorted(x for c in cmds for x in c['latency_ms'])
        diff = [g['served'] - runs[base][k]['served'] for k, g in games.items() if k in runs.get(base, {})]
        d = f'{sum(diff) / len(diff):+.2f} +- {se(diff):.2f}' if diff and name != base else '-'
        print(f'| {name} | {len(served)} | {sum(served) / len(served):.2f} +- {se(served):.2f} | {d} | {rate(cmds, lambda c: c["ok"])} | '
              f'{rate(amb, lambda c: c["asked"])} | {rate(clear, lambda c: c["asked"])} | {rate(cmds, lambda c: c["wait"])} | '
              + (f'{lat[len(lat) // 2]:.0f} / {lat[int(0.95 * (len(lat) - 1))]:.0f}' if lat else '-') + ' |')
    print()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--log', nargs='+', default=['runs/grpo-1.7b-v3/train.log', 'runs/grpo-0.6b/train.log'])
    a.add_argument('--results', default='results')
    a.add_argument('--window', type=int, default=25)
    args = a.parse_args()
    for log in args.log:
        if os.path.exists(log):
            trend(log, args.window)
    live(args.results)
    got = results(args.results)
    labels = [label for _, _, label in POLICIES]
    print('## Summary (turn-1 reward / ask F1)\n')
    print('| Policy | ' + ' | '.join(SPLITS) + ' |\n|---|' + '---|' * len(SPLITS))
    for label in labels:
        cells = [f"{got[label, sp]['turn-1 reward']:.2f} / {got[label, sp]['ask F1']:.2f}" if (label, sp) in got else '-'
                 for sp in SPLITS]
        print(f'| {label} | ' + ' | '.join(cells) + ' |')
    for sp in SPLITS:
        have = [label for label in labels if (label, sp) in got]
        if not have:
            continue
        print(f'\n## {sp}\n\n| Policy | ' + ' | '.join(SHOW.values()) + ' | Latency p50 / p95 ms |\n|---|' + '---|' * (len(SHOW) + 1))
        for label in have:
            r = got[label, sp]
            print(f'| {label} | ' + ' | '.join(f'{r[c]:.2f}' for c in SHOW) + f" | {r['latency'] or '-'} |")


if __name__ == '__main__':
    main()
