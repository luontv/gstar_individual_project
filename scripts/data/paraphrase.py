import argparse
import json
import re

from askoract import clarify as K
from askoract.backends import API

NAMES = re.compile(r'\b(tomato|lettuce|onion|alice|bob|cathy|david)s?\b', re.I)
PROMPT = """A player in a fast cooperative cooking game shouts short commands to their teammate.
Rewrite this command {n} different ways, the way real players talk: casual, short, sometimes clipped.
Keep the same meaning and the same level of vagueness: do not add which ingredient or soup is meant.
{slot}
Command: "{text}"
Answer with a JSON list of strings only."""


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', default='gpt-5.6-luna')
    a.add_argument('--n', type=int, default=8)
    a.add_argument('--out', default='data/paraphrases.json')
    args = a.parse_args()

    jobs = []  # (key, template index, text)
    held = set()
    for key, forms in K.TEMPLATES.items():
        for k, text in enumerate(forms):
            if len(forms) > 1 and k == len(forms) - 1:
                held.add(text.lower())
            else:
                jobs.append((key, k, text))
    slot = {True: 'Keep the placeholder {x} exactly once; it stands for the item.', False: ''}
    msgs = [[{'role': 'user', 'content': PROMPT.format(n=args.n, text=t, slot=slot['{x}' in t])}] for _, _, t in jobs]
    outs = API(args.model)(msgs)

    result, kept, dropped = {}, 0, 0
    for (key, k, text), o in zip(jobs, outs):
        try:
            cands = json.loads(re.search(r'\[.*\]', o, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            cands = []
        good = []
        for c in cands:
            c = str(c).strip().strip('"').rstrip('.').strip()
            # must stay exactly as vague: no ingredient or soup names, and {x} kept once in explicit patterns
            ok = (c.count('{x}') == 1 and not NAMES.search(c.replace('{x}', ''))) if '{x}' in text \
                else ('{x}' not in c and not NAMES.search(c))
            # also drop near copies of a held-out pattern ("do that one more time" vs held-out "one more time")
            near = any(h in c.lower() or h.replace('{x}', '').strip() in c.lower() for h in held)
            ok = ok and 0 < len(c) <= 60 and not near and c.lower() != text.lower()
            if ok and c not in good:
                good.append(c)
        kept += len(good)
        dropped += len(cands) - len(good)
        result[f'{key[0]}:{key[1]}:{k}'] = good
    json.dump(result, open(args.out, 'w'), indent=1)
    print(f'{len(jobs)} training patterns, kept {kept} rewordings, dropped {dropped}; saved {args.out}')


if __name__ == '__main__':
    main()
