import math
from collections import defaultdict

RECIPE = {'Alice': {'Lettuce', 'Onion'}, 'Bob': {'Lettuce', 'Tomato'},
          'Cathy': {'Onion', 'Tomato'}, 'David': {'Lettuce', 'Onion', 'Tomato'}}
INGREDIENTS = ('Tomato', 'Lettuce', 'Onion')
VERBS = ('Chop', 'Prepare', 'Cook', 'Plate', 'Serve')  # macros that take an object
NEAR = 2  # Manhattan distance that counts as "next to me"

TEMPLATES = {  # (verb, ref) -> phrasings, {x} = the object
    ('Chop', 'explicit'): ['chop {x}', 'can you chop some {x}', 'need {x} chopped'],
    ('Prepare', 'explicit'): ['prepare the {x} ingredients', 'get {x} ingredients ready'],
    ('Cook', 'explicit'): ['cook {x} soup', 'start the {x} soup'],
    ('Plate', 'explicit'): ['plate the {x} soup', 'put the {x} soup on a plate'],
    ('Serve', 'explicit'): ['serve {x} soup', 'deliver the {x} soup'],
    ('Chop', 'bare'): ['chop something', 'can you chop', 'we need veggies chopped'],
    ('Prepare', 'bare'): ['prepare some ingredients', 'get the ingredients ready'],
    ('Cook', 'bare'): ['cook a soup', 'start cooking'],
    ('Plate', 'bare'): ['plate the soup', 'grab a plate for the soup'],
    ('Serve', 'bare'): ['serve it', 'send a soup out'],
    ('Chop', 'other'): ['now chop the other one', 'the other veggie now'],
    ('Prepare', 'other'): ['prepare the other one', 'now the other ingredients'],
    ('Cook', 'other'): ['cook the other soup', 'now the other one'],
    ('Plate', 'other'): ['plate the other one'],
    ('Serve', 'other'): ['serve the other one'],
    (None, 'again'): ['do that again', 'same again', 'one more time'],
    ('Chop', 'urgent'): ['chop for the order running out', 'chop what the urgent order needs'],
    ('Prepare', 'urgent'): ['prepare the urgent one'],
    ('Cook', 'urgent'): ['cook the one about to expire', 'cook the urgent soup'],
    ('Plate', 'urgent'): ['plate the urgent one'],
    ('Serve', 'urgent'): ['serve the one about to expire', 'quick, serve the urgent one'],
    ('Chop', 'holding'): ['chop what goes with this', 'chop something for this'],
    ('Prepare', 'holding'): ['prepare the one that uses this'],
    ('Cook', 'holding'): ['cook the soup that goes with what I have'],
    ('Plate', 'near'): ['plate the soup next to me', 'plate the one over here'],
    ('Serve', 'near'): ['serve the one next to me', 'take this one out'],
}


def split(macro):
    verb, *rest = macro.split()
    return verb, (rest[0] if rest else None)


# HLA's four time buckets, 0 = most urgent
def urgency(rate):
    return 3 if rate > 0.75 else 2 if rate > 0.5 else 1 if rate > 0.25 else 0


# soup -> its most urgent bucket
def order_soups(s):
    out = {}
    for o in s['orders']:
        soup = o['name'].split()[1]
        out[soup] = min(out.get(soup, 9), urgency(o['rate']))
    return out


# higher = more urgent; only used to tell intent apart from strategy
def value(verb, obj, s):
    orders = order_soups(s)
    soups = [x for x in orders if obj in RECIPE[x]] if verb == 'Chop' else [obj] if obj in orders else []
    return max((-orders[x] for x in soups), default=-9)


def best(verb, objs, s):
    return max(objs, key=lambda o: value(verb, o, s))


def objects(verb, s):
    return [split(m)[1] for m in s['feasible'] if split(m)[0] == verb]


def relevant(verb, s):
    orders = order_soups(s)
    objs = objects(verb, s)
    if verb == 'Chop':
        need = set().union(*(RECIPE[x] for x in orders)) if orders else set()
        rel = [o for o in objs if o in need]
    else:
        rel = [o for o in objs if o in orders]
    return rel or objs


# e.g. 'Chopped Onion' -> ({'Onion'}, set())
def held_parts(item):
    words = set((item or '').split())
    return words & set(INGREDIENTS), words & set(RECIPE)


# objects the command could refer to, None if this reference type doesn't apply
def candidates(verb, ref, s):
    if ref == 'bare':
        return relevant(verb, s)
    if ref == 'other':
        last = [split(m)[1] for m in s['ai_hist'] if split(m)[0] == verb]
        if not last:
            return None
        return [o for o in relevant(verb, s) if o != last[-1]] or None
    if ref == 'urgent':
        rel = relevant(verb, s)
        if not rel or not s['orders']:
            return None
        top = max(value(verb, o, s) for o in rel)
        return [o for o in rel if value(verb, o, s) == top] if top > -9 else None
    if ref == 'holding':
        ings, soups = held_parts(s['human']['hold'])
        if verb == 'Chop':
            soups = order_soups(s) or RECIPE
            partners = set().union(*(RECIPE[x] - ings for x in soups if ings <= RECIPE[x])) if ings else set()
            c = [o for o in relevant(verb, s) if o in partners]
        else:
            c = [o for o in relevant(verb, s) if o in soups or (ings and ings <= RECIPE[o])]
        return c or None
    if ref == 'near':
        hx, hy = s['human']['pos']
        close = lambda p: abs(p[0] - hx) + abs(p[1] - hy) <= NEAR
        if verb == 'Plate':
            here = {p['soup'].split()[0] for p in s['pots'] if p['status'] == 'Cooked' and close(p['pos'])}
        else:
            here = {i['item'].split()[1] for i in s['items'] if i['item'].startswith('Plated') and close(i['pos'])}
        return [o for o in objects(verb, s) if o in here] or None
    raise ValueError(ref)


# simulated player

# an option can name the candidate itself or be one level up:
# a soup for a chop ("for Alice or Bob?"), an ingredient for a soup ("the onion one?")
def describes(option, verb, obj):
    if option == obj:
        return True
    if verb == 'Chop':
        return option in RECIPE and obj in RECIPE[option]
    return option in INGREDIENTS and obj in RECIPE and option in RECIPE[obj]


# what the player answers, given what they meant. They only answer what was asked:
# no options -> the object itself; one matching option -> that option; several -> a tuple ("both");
# none match -> 'NONE'; the question is about something else -> None
def answer(question, gold, cands, verb):
    opts = question.get('options')
    if opts is None:
        return gold
    if not any(describes(o, verb, c) for o in opts for c in cands):
        return None
    hits = tuple(o for o in opts if describes(o, verb, gold))
    return hits[0] if len(hits) == 1 else hits or 'NONE'


# expected information gain over a uniform prior on cands, 0 to 1
def info_gain(question, cands, verb):
    if len(cands) < 2:
        return 0.0
    groups = defaultdict(int)
    for g in cands:
        groups[answer(question, g, cands, verb)] += 1
    n = len(cands)
    h_after = sum(k / n * math.log(k) for k in groups.values())
    return 1 - h_after / math.log(n)


# candidates consistent with the reply
def remaining(question, reply, cands, verb):
    return [c for c in cands if answer(question, c, cands, verb) == reply]


# what a scripted AI does after the reply: the one candidate left, else the most urgent of those left
def settle(question, value, cands, verb, s):
    left = cands if value in (None, 'ANY') else remaining(question, value, cands, verb)
    return left[0] if len(left) == 1 else best(verb, left, s)


REPLY_FORMS = {
    'direct': ['{x}', '{x} please', 'the {x} one'],
    'negation': ['not {y}', 'no, not {y}'],  # only with two options: say which one you don't want
    'indifferent': ['either, just hurry', "doesn't matter"],
}


# returns (reply text, value the AI can use, objects that count as correct).
# value is None for silence or a useless reply, 'ANY' when the player doesn't care
def reply(question, gold, cands, verb, rng, p_indifferent=0.1, p_silent=0.1, p_negation=0.2):
    r = rng.random()
    if r < p_silent:
        return '', None, [gold]
    if r < p_silent + p_indifferent:
        return rng.choice(REPLY_FORMS['indifferent']), 'ANY', list(cands)
    value = answer(question, gold, cands, verb)
    if value is None:
        return rng.choice(['what?', 'just do it']), None, [gold]
    if value == 'NONE':
        return 'neither', 'NONE', [gold]
    if isinstance(value, tuple):
        return rng.choice(['both', 'all of those', 'the one with both']), value, [gold]
    opts = question.get('options') or cands
    if len(opts) == 2 and rng.random() < p_negation:
        other = next(o for o in opts if o != value)
        return rng.choice(REPLY_FORMS['negation']).format(y=other.lower()), value, [gold]
    return rng.choice(REPLY_FORMS['direct']).format(x=value.lower()), value, [gold]


def say(options, verb):
    words = [o if o in RECIPE else o.lower() for o in options]
    if verb == 'Chop' and options[0] in RECIPE:
        return f"For {' or '.join(words)}?"
    if verb != 'Chop' and options[0] in INGREDIENTS:
        return f"The {' one or the '.join(words)} one?"
    body = ', '.join(words[:-1]) + ' or ' + words[-1]
    return f"{body[0].upper()}{body[1:]}{' soup' if verb != 'Chop' else ''}?"


# command generation

def macro(verb, obj):
    return {'Chop': f'Chop {obj}', 'Prepare': f'Prepare {obj} Ingredients', 'Cook': f'Cook {obj} Soup',
            'Plate': f'Plate {obj} Soup', 'Serve': f'Serve {obj} Soup'}[verb]


# one command per (state, verb, reference type) that applies.
# all_golds: one copy per possible intent for ambiguous commands
def episodes(states, rng, all_golds=False):
    for sid, s in enumerate(states):
        for verb in VERBS:
            objs = objects(verb, s)
            if not objs:
                continue
            refs = [('explicit', [rng.choice(objs)])]
            for ref in ('bare', 'other', 'urgent', 'holding', 'near'):
                if (verb, ref) in TEMPLATES:
                    c = candidates(verb, ref, s)
                    if c:
                        refs.append((ref, c))
            for ref, cands in refs:
                k = rng.randrange(len(TEMPLATES[(verb, ref)]))
                golds = cands if all_golds else [rng.choice(cands)]  # intent doesn't depend on urgency
                for gold in golds:
                    yield make(sid, s, verb, ref, k, cands, gold)
        if s['ai_hist'] and s['ai_hist'][-1] in s['feasible'] and split(s['ai_hist'][-1])[1]:
            verb, obj = split(s['ai_hist'][-1])
            if verb in VERBS:
                k = rng.randrange(len(TEMPLATES[(None, 'again')]))
                yield make(sid, s, verb, 'again', k, [obj], obj)


def make(sid, s, verb, ref, k, cands, gold):
    key = (None, 'again') if ref == 'again' else (verb, ref)
    vals = {c: value(verb, c, s) for c in cands}
    top = max(vals.values())
    return {
        'state_id': sid, 'game': s['game'], 'map': s['map'],
        'command': TEMPLATES[key][k].format(x=gold.lower()),
        'family': f'{key[0]}:{ref}:{k}', 'verb': verb, 'ref': ref,
        'cands': cands, 'gold': gold, 'gold_macro': macro(verb, gold),
        'ambiguous': len(cands) > 1,
        # clear only thanks to context (several options, but history/orders/position pick one)
        'context_resolved': len(cands) == 1 and ref != 'explicit' and len(objects(verb, s)) > 1,
        'gold_is_best': vals[gold] == top and list(vals.values()).count(top) == 1,
        'value_conflict': vals[gold] < top,
    }


if __name__ == '__main__':
    c = ['Tomato', 'Lettuce', 'Onion']
    ig = lambda opts, cands, verb='Chop': info_gain({'options': opts}, cands, verb)
    assert ig(c, c) == 1.0                                   # "tomato, lettuce or onion?"
    assert ig(None, c) == 1.0                                # "which one?"
    assert ig(['Plate'], c) == 0.0                           # asks about something else
    assert 0 < ig(['Tomato'], c) < 1                         # "tomato?" narrows only
    assert ig(['Tomato'], c[:2]) == 1.0                      # yes/no settles two options
    assert ig(['Alice', 'Bob'], ['Tomato', 'Onion']) == 1.0  # soup-level question for a chop command
    assert ig(['Lettuce'], ['Alice', 'Bob'], 'Cook') == 0.0  # both soups use lettuce
    assert ig(['Onion', 'Tomato'], ['Alice', 'Bob'], 'Cook') == 1.0
    assert answer({'options': ['Onion', 'Tomato']}, 'David', ['Bob', 'David'], 'Cook') == ('Onion', 'Tomato')
    assert remaining({'options': ['Tomato']}, 'NONE', c, 'Chop') == ['Lettuce', 'Onion']
    s = {'orders': [{'name': 'Plated Cathy Soup', 'rate': 0.9}, {'name': 'Plated Bob Soup', 'rate': 0.1}],
         'feasible': ['Chop Tomato', 'Chop Lettuce', 'Chop Onion', 'Cook Bob Soup', 'Cook Cathy Soup'],
         'ai_hist': ['Chop Tomato'], 'human': {'pos': (0, 0), 'hold': 'Chopped Onion'}, 'pots': [], 'items': []}
    assert candidates('Chop', 'bare', s) == ['Tomato', 'Lettuce', 'Onion']
    assert candidates('Chop', 'other', s) == ['Lettuce', 'Onion']
    assert candidates('Chop', 'urgent', s) == ['Tomato', 'Lettuce']  # Bob is about to expire
    assert candidates('Cook', 'holding', s) == ['Cathy']             # onion -> Cathy, not Bob
    assert value('Cook', 'Bob', s) > value('Cook', 'Cathy', s)
    print('ok')
