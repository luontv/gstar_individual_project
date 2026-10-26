import json

from askoract.prompt import parse

MODELS = {'gemini-3.5-flash': 'api_gemini_3_5_flash', 'qwen/qwen3.8-flash': 'api_qwen3_8_flash',
          'deepseek-v4-flash': 'api_deepseek_v4_flash'}


# ambiguous commands a model labels as ambiguous (labeller check), and how often the same model asks on them when playing
def main():
    rows = [r['meta'] for r in map(json.loads, open('data/test/test_phrasing_nc.jsonl')) if r['meta']['kind'] == 'turn1']
    idx = {(e['state_id'], e['command']): i for i, e in enumerate(rows)}
    labels = [json.loads(l) for l in open('results/labeller_check.jsonl')]
    for name, prefix in MODELS.items():
        outs = [json.loads(l)['turn1'] for l in open(f'results/{prefix}_test_phrasing.jsonl')]
        seen = [(r, outs[idx[r['meta']['state_id'], r['meta']['command']]]) for r in labels
                if r['meta']['ambiguous'] and r.get(name) is not None]
        seen = [(r, o) for r, o in seen if not o.startswith('ERROR')]
        said = [o for r, o in seen if len(r[name]) > 1]
        asked = sum(parse(o).get('mode') == 'ask' for o in said)
        print(f'{name:20s} ambiguous {len(seen)}  labelled ambiguous {len(said)}  asked when playing {asked}')


if __name__ == '__main__':
    main()
