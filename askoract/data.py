import json

from askoract import clarify as K

TEST_MAP = 'partition'
VAL_GAME = 4


def execute(m):
    return {'mode': 'execute', 'macro': m}


def turn1(e):
    if e['ambiguous']:
        return {'mode': 'ask', 'options': e['cands'], 'question': K.say(e['cands'], e['verb'])}
    return execute(e['gold_macro'])


def heldout(e):
    key = (None, 'again') if e['ref'] == 'again' else (e['verb'], e['ref'])
    n = len(K.TEMPLATES[key])
    return n > 1 and int(e['family'].split(':')[-1]) == n - 1


def split_of(e, s):
    if s['map'] == TEST_MAP:
        return 'test_map'
    if int(s['game'].split('-')[1]) == VAL_GAME:
        return 'test_phrasing' if heldout(e) else 'val'
    return None if heldout(e) else 'train'


def row(msgs, target, meta):
    return {'messages': msgs + [{'role': 'assistant', 'content': json.dumps(target)}], 'meta': meta}
