import json
import re

LABEL = """In a cooperative cooking game, the human player sent a chat message to their AI teammate.
Recipes: Alice = lettuce + onion; Bob = lettuce + tomato; Cathy = onion + tomato; David = lettuce + onion + tomato.

Situation (from the teammate's view):
{situation}

Player's message: "{message}"

List every action from "Possible actions now" that the player might mean by this message.
Give a single action only if a reasonable teammate would be confident that it is the one meant; if a teammate could not
be sure between several, list all of them (do not just pick the most useful or most urgent one).
Answer with a JSON list of action names only."""


# Cohen's kappa for two yes/no raters
def kappa(pairs):
    po = sum(a == b for a, b in pairs) / len(pairs)
    pa, pb = (sum(p[i] for p in pairs) / len(pairs) for i in (0, 1))
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def macros(text, feasible):
    try:
        got = json.loads(re.search(r'\[.*\]', text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return None
    return [m for m in feasible if m in set(map(str, got))]
