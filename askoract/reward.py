from askoract import clarify as K
from askoract.prompt import acted_obj, parse, question

ASK_COST = 0.2       # time a question costs in a real-time game
OVER_ASK = -0.5      # asking when the command was already clear
WRONG = -1.0         # acting on an ambiguous command, or acting wrongly on a clear one


def turn1_reward(e, asked, ig, final):
    if asked:
        return ig - ASK_COST if e['ambiguous'] else OVER_ASK
    return WRONG if e['ambiguous'] or final != e['gold'] else 1.0


# (reward, asked) for one model reply
def reply_reward(text, e):
    out = parse(text)
    if out.get('mode') == 'ask':
        q = question(out)
        return turn1_reward(e, True, K.info_gain(q, e['cands'], e['verb']), None), True
    return turn1_reward(e, False, 0.0, acted_obj(out, e['verb'])), False


def summarize(name, recs):
    n = len(recs)
    amb = [r for r in recs if r['e']['ambiguous']]
    asked = [r for r in recs if r['asked']]
    asked_amb = [r for r in amb if r['asked']]
    clear = n - len(amb)
    ctx = [r for r in recs if r['e']['context_resolved']]
    conflict = [r for r in amb if r['e']['value_conflict']]
    isbest = [r for r in amb if r['e']['gold_is_best']]
    mean = lambda xs, f: sum(map(f, xs)) / len(xs) if xs else 0.0
    p = len(asked_amb) / len(asked) if asked else 0
    rcl = len(asked_amb) / len(amb) if amb else 0
    return {
        'policy': name,
        'intent acc': mean(recs, lambda r: r['ok']),
        'acc | gold best': mean(isbest, lambda r: r['ok']),
        'acc | gold not best': mean(conflict, lambda r: r['ok']),
        'ask P': p, 'ask R': rcl, 'ask F1': 2 * p * rcl / (p + rcl) if p + rcl else 0,
        'over-ask': (len(asked) - len(asked_amb)) / clear if clear else 0,
        'over-ask | context': mean(ctx, lambda r: r['asked']),
        'question IG': mean(asked_amb, lambda r: r['ig']),
        'turn-1 reward': mean(recs, lambda r: r['reward']),
    }


def table(rows):
    keys = list(rows[0])
    out = ['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
    for r in rows:
        out.append('| ' + ' | '.join(v if isinstance(v, str) else f'{v:.2f}' for v in r.values()) + ' |')
    return '\n'.join(out)
