import json
import re

from askoract import clarify as K

OBJS = set(K.INGREDIENTS) | set(K.RECIPE)  # names a reply may refer to
BUCKET = ['expiring', 'little time', 'some time', 'plenty of time']
SYSTEM = """You are the AI chef (blue) in a cooperative cooking game. The human player (pink) gives you short commands while playing.
Reply with ONE JSON object:
- {"mode": "execute", "macro": "<action>"} when the command clearly means one action.
- {"mode": "ask", "options": [...], "question": "..."} when it could mean several possible actions and the situation does not settle it. Questions cost time, so ask only when needed.
- {"mode": "refuse", "reason": "..."} when the requested action is not possible right now.
- {"mode": "reply", "text": "..."} when the message needs no action.
Use the situation: your last actions ("the other one"), the order running out ("the urgent one"), what the player holds ("this"), what is next to the player ("next to me").
Recipes: Alice = lettuce + onion; Bob = lettuce + tomato; Cathy = onion + tomato; David = lettuce + onion + tomato.
Actions: Chop <Tomato|Lettuce|Onion>, Prepare <Soup> Ingredients, Cook <Soup> Soup, Plate <Soup> Soup, Serve <Soup> Soup, Putout, Drop."""
COORDS = False  # runs 1-2 put grid coordinates in the prompt, run 3 onwards doesn't


# situation text; without coordinates, things near the player get tagged instead
def observe(s, coords=None):
    coords = COORDS if coords is None else coords
    orders = ', '.join(f"{o['name'].replace('Plated ', '')} ({BUCKET[K.urgency(o['rate'])]})" for o in s['orders'])
    state = {'Cooking': 'cooking', 'Cooked': 'cooked', 'Fire': 'on fire with', 'Charred': 'charred'}
    pot = lambda p: 'empty' if p['soup'] is None else f"{state.get(p['status'], '')} {p['soup']}"
    item = lambda i: i['item'] + (' (cutboard)' if i['on'] == 'Cutboard' else '')
    if coords:
        pots = '; '.join(f"{tuple(p['pos'])} {pot(p)}" for p in s['pots'])
        items = '; '.join(f"{i['item']} at {tuple(i['pos'])}" + (' (cutboard)' if i['on'] == 'Cutboard' else '')
                          for i in s['items'])
        you = f"You are at {tuple(s['ai']['pos'])} holding {s['ai']['hold'] or 'nothing'}.\n"
        player = f"Player is at {tuple(s['human']['pos'])} holding {s['human']['hold'] or 'nothing'}.\n"
    else:
        hx, hy = s['human']['pos']

        def group(things, name):
            counts = {}
            for t in things:
                near = abs(t['pos'][0] - hx) + abs(t['pos'][1] - hy) <= K.NEAR
                key = name(t) + (' (next to the player)' if near else '')
                counts[key] = counts.get(key, 0) + 1
            return '; '.join(f'{n} x {k}' if n > 1 else k for k, n in counts.items())
        pots, items = group(s['pots'], pot), group(s['items'], item)
        you = f"You hold {s['ai']['hold'] or 'nothing'}.\n"
        player = f"Player holds {s['human']['hold'] or 'nothing'}.\n"
    return (f"Orders: {orders or 'none'}\n" + you + player +
            f"Pots: {pots or 'none'}\n"
            f"On counters: {items or 'nothing'}\n"
            f"Your last actions: {', '.join(s['ai_hist']) or 'none'}\n"
            f"Possible actions now: {', '.join(s['feasible']) or 'none'}")


def messages(s, command, asked=None, replied=None):
    user = observe(s) + f'\nPlayer says: "{command}"'
    if asked is not None:
        user += f'\nYou asked: "{asked}"\nPlayer replied: "{replied}"'
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


# same prompt as at inference: chat template + generation prompt, thinking off
def prompt_text(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def to_obj(x):
    return next((w.capitalize() for w in re.findall(r'[A-Za-z]+', str(x)) if w.capitalize() in OBJS), None)


# the question in an ask reply; options we can't parse count as a useless question
def question(out):
    return {'options': [x for x in map(to_obj, out.get('options') or []) if x]}


# object the reply acts on, None if it isn't an execute for this verb
def acted_obj(out, verb):
    if out.get('mode') != 'execute':
        return None
    v, obj = K.split(str(out.get('macro', 'x')))
    return obj if v == verb else None
