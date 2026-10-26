import json
import random
import re

from askoract import clarify as K
from askoract.data import TEST_MAP, VAL_GAME, row, turn1
from askoract.policies import REFS, VAGUE
from askoract.prompt import messages
from scripts.eval.uncertainty import VERB_WORDS

# new phrasings for every command meaning, written after training and kept out of it: none uses a word from the
# rules' keyword lists, the reference hints in the model prompt, or the training / held-out templates
NEW = {
    ('Chop', 'explicit'): ['dice the {x}', 'slice up a {x}'],
    ('Prepare', 'explicit'): ['put together the {x} mix', 'gather what {x} needs'],
    ('Cook', 'explicit'): ['get the {x} boiling', 'put {x} on the stove'],
    ('Plate', 'explicit'): ['dish up the {x}', 'scoop {x} into a bowl'],
    ('Serve', 'explicit'): ['hand in the {x}', 'take {x} to the window'],
    ('Chop', 'bare'): ['dice whatever we need', 'get some cutting done'],
    ('Prepare', 'bare'): ['put a mix together', 'gather stuff for an order'],
    ('Cook', 'bare'): ['get a pot boiling', 'fire up the stove'],
    ('Plate', 'bare'): ['dish one up', 'fill a bowl'],
    ('Serve', 'bare'): ['hand one in', 'take a finished bowl to the window'],
    ('Chop', 'other'): ['dice a different one now', 'now cut a different kind'],
    ('Prepare', 'other'): ['now put together a different mix', 'gather for a different order now'],
    ('Cook', 'other'): ['boil a different one', 'now put a different one on the stove'],
    ('Plate', 'other'): ['dish up a different one'],
    ('Serve', 'other'): ['hand in a different one'],
    (None, 'again'): ['repeat that', 'same as before', 'do that once more'],
    ('Chop', 'urgent'): ['dice for the order that is nearly gone', 'cut what the most pressing order needs'],
    ('Prepare', 'urgent'): ['put together the most pressing order', 'gather for the order closest to timing out'],
    ('Cook', 'urgent'): ['boil whichever order is nearly gone', 'get the most pressing order on the stove'],
    ('Plate', 'urgent'): ['dish up the most pressing one'],
    ('Serve', 'urgent'): ['hand in whatever is closest to timing out', 'hurry, hand in the most pressing one'],
    ('Chop', 'holding'): ['dice what pairs with mine', 'cut whatever matches what I am carrying'],
    ('Prepare', 'holding'): ['gather for the order my item belongs to'],
    ('Cook', 'holding'): ['boil the one my item belongs to'],
    ('Plate', 'near'): ['dish up the pot beside me', 'fill a bowl from the pot by my side'],
    ('Serve', 'near'): ['hand in the bowl beside me', 'take the dish by my side to the window'],
}
HINTS = re.compile(r'the other one|the urgent one|next to me|\bthis\b|again|one more time|quick|grab|ready|over here')
OLD = {t for ts in K.TEMPLATES.values() for t in ts}


def check():
    assert set(NEW) == set(K.TEMPLATES)
    names = re.compile(r'\b(' + '|'.join(o.lower() for o in (*K.INGREDIENTS, *K.RECIPE)) + r')\b')
    for t in (t for ts in NEW.values() for t in ts):
        assert t not in OLD and not (VAGUE.search(t) or HINTS.search(t) or names.search(t.replace('{x}', '')) or
                                     any(re.search(w, t) for _, w in REFS) or any(re.search(w, t) for _, w in VERB_WORDS)), t


# same game moments and sampling as test_phrasing (validation game, training maps), every intent, the new phrasings
def main(n=2000, seed=0):
    check()
    states = [json.loads(l) for l in open('data/states_v2.jsonl')]
    K.TEMPLATES = NEW  # episodes() and make() read the module's templates
    keep = ('state_id', 'command', 'verb', 'ref', 'family', 'cands', 'gold', 'gold_macro', 'ambiguous', 'context_resolved',
            'gold_is_best', 'value_conflict')
    rows = [row(messages(states[e['state_id']], e['command']), turn1(e), {k: e[k] for k in keep} | {'kind': 'turn1'})
            for e in K.episodes(states, random.Random(seed), all_golds=True)
            if e['map'] != TEST_MAP and int(e['game'].split('-')[1]) == VAL_GAME]
    rng = random.Random(seed)
    rng.shuffle(rows)
    amb = [r for r in rows if r['meta']['ambiguous']][:n // 2]  # at most half ambiguous, as in the other test splits
    rows = amb + [r for r in rows if not r['meta']['ambiguous']][:n - len(amb)]
    rng.shuffle(rows)
    with open('data/test/test_new_nc.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    m = [r['meta'] for r in rows]
    print(f"{len(m)} rows, {len({e['state_id'] for e in m})} moments, {sum(e['ambiguous'] for e in m)} ambiguous, "
          f"{sum(e['context_resolved'] for e in m)} clear from context, {len({e['command'] for e in m})} distinct commands")


if __name__ == '__main__':
    main()
