import re

from askoract import clarify as K


def never_ask_best(e, s, rng):
    return 'act', K.best(e['verb'], e['cands'], s)


def never_ask_random(e, s, rng):
    return 'act', rng.choice(e['cands'])


def always_ask(e, s, rng):
    return 'ask', e['cands']


# knows the true candidates, asks exactly when there's more than one
def oracle_rule(e, s, rng):
    return ('ask', e['cands']) if e['ambiguous'] else ('act', e['cands'][0])


# words from the training commands only; none from the held-out wording
VAGUE = re.compile(r'\b(something|other|this|it|urgent|expire|running out|next to me|again|soup|ingredients)\b')
AGAIN = re.compile(r'\b(again|one more time)\b')


# act on a named object, or repeat the last action on "again"; None if the command does neither
def named_or_again(e, s, objs):
    named = [o for o in objs if o.lower() in e['command']]
    if named:
        return 'act', named[0]
    if AGAIN.search(e['command']) and s['ai_hist']:
        return 'act', K.split(s['ai_hist'][-1])[1]
    return None


# keyword rule: act on a named object, repeat on "again", ask if the wording is vague
def text_rule(e, s, rng):
    objs = K.objects(e['verb'], s)
    hit = named_or_again(e, s, objs)
    if hit:
        return hit
    if len(objs) > 1 and VAGUE.search(e['command']):
        return 'ask', objs
    return 'act', objs[0]


# words that mark each reference type, from the training commands only
REFS = [('near', r'next to me'), ('holding', r'goes with|uses this|for this|what i have'),
        ('urgent', r'urgent|expire|running out'), ('other', r'\bother\b')]


# keyword + state rule: reference type from keywords, its candidates from the game state, ask if more than one
def state_rule(e, s, rng):
    hit = named_or_again(e, s, K.objects(e['verb'], s))
    if hit:
        return hit
    ref = next((r for r, w in REFS if re.search(w, e['command'])), 'bare')
    c = K.candidates(e['verb'], ref, s) or K.relevant(e['verb'], s)
    return ('ask', c) if len(c) > 1 else ('act', c[0])


POLICIES = [never_ask_random, never_ask_best, always_ask, text_rule, state_rule, oracle_rule]
