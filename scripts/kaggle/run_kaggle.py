import os
import subprocess
import sys
import time

OUT = '/kaggle/working/out'
STEPS = 312  # 312 x 16 = 4,992 of the 5,000 commands, i.e. one pass
SPLITS = ('val', 'test_phrasing', 'test_map', 'test_free')


def evals(gpu, name, adapter=None, splits=SPLITS, model='Qwen/Qwen3-1.7B'):
    adapter = f'--adapter {adapter} ' if adapter else ''
    return '; '.join(f'CUDA_VISIBLE_DEVICES={gpu} python -m scripts.eval.eval_model --model {model} {adapter}--split data/test/{s}_nc.jsonl '
                     f'--states data/states_v2.jsonl --save out/{name}_{s}.jsonl > out/{name}_{s}.txt 2>&1' for s in splits)


def sh(cmd):
    print('$', cmd[:200], flush=True)
    subprocess.run(cmd, shell=True)


def sft(gpu, flags, out, log):
    return (f'CUDA_VISIBLE_DEVICES={gpu} python -u -m scripts.train.train_sft {flags} --out {out} > out/{log}.log 2>&1; '
            f'rm -rf {out}/checkpoint-*; ')


# one job set per notebook version, both GPUs in parallel (outputs in /kaggle/working/out):
#   grpo:    GRPO from SFT run 3 | same first turns with plain SFT (control)
#   control: full SFT recipe on the same commands | first turns only at lr 2e-5
#   small:   Qwen3-0.6B SFT | untrained 0.6B on all splits, untrained 1.7B on val
#   seeds:   Qwen3-0.6B SFT with --seed 1 | --seed 2
#   grpo06:  Qwen3-0.6B SFT (--seed 42), then GRPO on fresh commands (data/rl2) | the SFT adapter evaluated, then
#            the full SFT recipe continued on data/rl2 (control); data/rl2 is built in main() with --seed 2
RUN3 = '--init runs/sft-1.7b-v3'
SMALL = '--model Qwen/Qwen3-0.6B --data data/rl/train.jsonl'
JOBS = {'grpo': [
    f'CUDA_VISIBLE_DEVICES=0 python -u -m scripts.train.train_grpo --steps {STEPS} --micro 4 --out out/grpo-1.7b-v3 > out/grpo.log 2>&1; '
    + evals(0, 'grpo', 'out/grpo-1.7b-v3'),
    sft(1, f'{RUN3} --data data/rl/turn1.jsonl', 'out/sftmore-1.7b-v3', 'sftmore')
    + evals(1, 'sftmore', 'out/sftmore-1.7b-v3') + '; ' + evals(1, 'v3', 'runs/sft-1.7b-v3', ('val',)),
], 'control': [
    sft(0, f'{RUN3} --data data/rl/train.jsonl', 'out/sftall-1.7b-v3', 'sftall') + evals(0, 'sftall', 'out/sftall-1.7b-v3'),
    sft(1, f'{RUN3} --data data/rl/turn1.jsonl --lr 2e-5', 'out/sftlow-1.7b-v3', 'sftlow') + evals(1, 'sftlow', 'out/sftlow-1.7b-v3'),
], 'small': [
    sft(0, SMALL, 'out/sft-0.6b', 'sft06') + evals(0, 'sft06', 'out/sft-0.6b', model='Qwen/Qwen3-0.6B'),
    evals(1, 'base06', model='Qwen/Qwen3-0.6B') + '; ' + evals(1, 'base', splits=('val',)),
], 'seeds': [
    sft(g, f'{SMALL} --seed {g + 1}', f'out/sft06-s{g + 1}', f'sft06-s{g + 1}')
    + evals(g, f'sft06s{g + 1}', f'out/sft06-s{g + 1}', model='Qwen/Qwen3-0.6B') for g in (0, 1)
], 'grpo06': [
    sft(0, f'{SMALL} --seed 42', 'out/sft06g', 'sft06g') + 'touch out/sft06g.done; '
    f'CUDA_VISIBLE_DEVICES=0 python -u -m scripts.train.train_grpo --model Qwen/Qwen3-0.6B --init out/sft06g '
    f'--data data/rl2/train.jsonl --steps {STEPS} --out out/grpo06 > out/grpo06.log 2>&1; '
    + evals(0, 'grpo06', 'out/grpo06', model='Qwen/Qwen3-0.6B'),
    'while [ ! -f out/sft06g.done ]; do sleep 30; done; ' + evals(1, 'sft06g', 'out/sft06g', model='Qwen/Qwen3-0.6B') + '; '
    + sft(1, '--model Qwen/Qwen3-0.6B --init out/sft06g --data data/rl2/train.jsonl', 'out/sftall06', 'sftall06')
    + evals(1, 'sftall06', 'out/sftall06', model='Qwen/Qwen3-0.6B'),
]}
PROGRESS = {'grpo': ('grpo.log', 'sftmore.log'), 'control': ('sftall.log', 'sftlow.log'),
            'small': ('sft06.log', 'base06_test_free.txt'), 'seeds': ('sft06-s1.log', 'sft06-s2.log'),
            'grpo06': ('sft06g.log', 'grpo06.log', 'sftall06.log')}  # last line of each is printed every 2 min

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'grpo'
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists('out'):
        os.symlink(OUT, 'out')
    sh('pip uninstall -y -q torchao; pip install -q transformers==5.20.2 peft==0.21.2 2>&1 | tail -2')
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv; python -c "import torch, transformers, peft; '
       'print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.device_count())"')
    sh("grep '\"kind\": \"turn1\"' data/rl/train.jsonl > data/rl/turn1.jsonl; wc -l data/rl/turn1.jsonl")
    if which == 'grpo06':
        sh('python -m scripts.data.build_data --states data/states_v2.jsonl --para data/paraphrases.json --perturb 1 --drop 1 '
           '--seed 2 --out data/rl2 > out/build_rl2.log 2>&1; tail -4 out/build_rl2.log')
    procs = [subprocess.Popen(j, shell=True) for j in JOBS[which]]
    t = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        for log in PROGRESS[which]:
            try:
                last = open(f'{OUT}/{log}').read().replace('\r', '\n').strip().splitlines()[-1]
            except (OSError, IndexError):
                last = '-'
            print(f'[{(time.time() - t) / 60:.0f} min] {log}: {last[:160]}', flush=True)
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.txt'):
            print(f'\n== {f}\n' + ''.join(open(f'{OUT}/{f}').readlines()[-8:]), flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
