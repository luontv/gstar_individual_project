# Ask or Act: training a small game companion that knows when to clarify

A teammate AI in HLA's Overcooked testbed gets short, often vague commands from the human player and has to either act or
ask a clarifying question. What a command can mean is computed exactly from the game state, a simulated player answers
questions, and questions are scored by how much they narrow the options, so no LLM judge is needed. Qwen3 models (0.6B,
1.7B) are trained with LoRA SFT and GRPO and compared with rule baselines and a prompted API model, offline and in live games.

## Layout
```
askoract/            shared code
  clarify.py         candidate actions, simulated player, question score
  game.py            testbed glue: game moments, scripted macro picker
  prompt.py          model prompt and reply parsing
  data.py            splits and training targets
  reward.py          turn-1 reward and metrics
  policies.py        rule baselines
  backends.py        Hugging Face / LiteLLM / local-server models
  labelling.py       LLM labelling prompt and agreement
scripts/
  data/              sample_states, build_data, paraphrase, freeform, check_labellers,
                     build_new_wording (keyword-free test set, test_new)
  train/             train_sft, train_grpo
  eval/              eval_model, baselines, live, latency_mlx, report, uncertainty (95% intervals)
  kaggle/            run_kaggle (training and evaluation jobs on a 2x T4 notebook)
hla/                 HLA testbed (submodule)
```
Run everything from the repository root as modules, e.g. `python -m scripts.eval.live --policy keyword`.

## Setup
`git clone --recursive`, or `git submodule update --init` after cloning.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -e hla/testbed-cooking -e hla/agent
```
Training and evaluation use a second environment with torch, transformers 5.20.2 and peft 0.21.2 (a CUDA GPU; a T4 is
enough), and mlx-lm for latency and live games on a Mac. API models go through LiteLLM: put `LITELLM_API_KEY` and
`LITELLM_BASE_URL` in a `.env` file in the folder above this one.

Included: the frozen test sets (`data/test/`, `*_nc` = prompt without grid coordinates), the game moments
(`data/states*.jsonl`) and the training rewordings (`data/paraphrases.json`); training sets can be rebuilt from these with
`build_data`. `results/` holds the raw outputs and live-game logs behind every table (`python -m scripts.eval.report`;
intervals: `python -m scripts.eval.uncertainty`).
Trained adapters are not included.

## Example
```bash
python -m scripts.data.sample_states --games 5 --out data/states.jsonl
python -m scripts.data.build_data --states data/states.jsonl --para data/paraphrases.json --perturb 1 --drop 1 --out data/sft_v3
python -m scripts.train.train_sft --model Qwen/Qwen3-0.6B --data data/sft_v3/train.jsonl --out runs/sft-0.6b
python -m scripts.eval.eval_model --model Qwen/Qwen3-0.6B --adapter runs/sft-0.6b --split data/sft_v3/test_phrasing.jsonl --states data/states.jsonl
python -m scripts.eval.live --policy keyword --out results/live_keyword.jsonl
python -m scripts.eval.live --policy keyword --wording keyword-free --games 50 --out results/live_kf_keyword.jsonl
python -m scripts.eval.report --window 50 > results/report.md
```
