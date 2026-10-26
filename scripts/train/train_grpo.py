import argparse
import json
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from askoract.prompt import prompt_text
from askoract.reward import reply_reward


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', default='Qwen/Qwen3-1.7B')
    a.add_argument('--init', default='runs/sft-1.7b-v3', help='SFT adapter to start from')
    a.add_argument('--data', default='data/rl/train.jsonl', help='first-turn rows; only the meta is used')
    a.add_argument('--out', default='runs/grpo-1.7b-v3')
    a.add_argument('--steps', type=int, default=200)
    a.add_argument('--prompts', type=int, default=16, help='commands per step')
    a.add_argument('--group', type=int, default=8, help='samples per command')
    a.add_argument('--temp', type=float, default=1.0)
    a.add_argument('--lr', type=float, default=2e-5)
    a.add_argument('--gen-batch', type=int, default=8, help='commands per generate call')
    a.add_argument('--micro', type=int, default=8, help='sequences per forward/backward pass')
    a.add_argument('--save-steps', type=int, default=25)
    a.add_argument('--seed', type=int, default=0)
    args = a.parse_args()
    torch.manual_seed(args.seed)
    rows = [r for r in map(json.loads, open(args.data)) if r['meta']['kind'] == 'turn1']
    random.Random(args.seed).shuffle(rows)
    assert args.steps * args.prompts <= len(rows), 'not enough commands for one pass'

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = 'left'
    eos = tok.eos_token_id
    dev = 'cuda' if torch.cuda.is_available() else 'mps'  # mps = quick test on a Mac (fp32)
    dtype = torch.float32 if dev != 'cuda' else torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
        else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation='sdpa',
                                                 device_map={'': 0} if dev == 'cuda' else None).to(dev)  # mps device_map hangs
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0  # the loss must score the same policy that sampled
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    # no KL penalty to the SFT policy (as in Dr. GRPO / DAPO); a small lr and a single pass keep it close
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    print(f'{len(rows)} commands; {args.steps} steps x {args.prompts} commands x {args.group} samples; '
          f'{sum(p.numel() for p in params) / 1e6:.1f}M trainable', flush=True)

    t0 = time.time()
    for step in range(args.steps):
        batch = rows[step * args.prompts:(step + 1) * args.prompts]
        model.eval()
        samples = []
        for i in range(0, len(batch), args.gen_batch):
            chunk = batch[i:i + args.gen_batch]
            enc = tok([prompt_text(tok, r['messages'][:-1]) for r in chunk], return_tensors='pt', padding=True,
                      add_special_tokens=False).to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=dtype, enabled=dev == 'cuda'):
                g = model.generate(**enc, do_sample=True, temperature=args.temp, top_p=1.0, top_k=0, max_new_tokens=64,
                                   num_return_sequences=args.group, pad_token_id=tok.pad_token_id)
            for k, c in enumerate(g[:, enc['input_ids'].shape[1]:].tolist()):
                j = k // args.group  # generate repeats each prompt group times, in order
                c = c[:c.index(eos) + 1] if eos in c else c
                r, asked = reply_reward(tok.decode(c, skip_special_tokens=True), chunk[j]['meta'])
                samples.append({'g': i + j, 'p': enc['input_ids'][j][enc['attention_mask'][j].bool()].tolist(),
                                'c': c, 'r': r, 'asked': asked})
        groups = {}
        for s in samples:
            groups.setdefault(s['g'], []).append(s)
        for grp in groups.values():
            mean = sum(s['r'] for s in grp) / len(grp)  # Dr. GRPO: reward minus group mean, no std scaling
            for s in grp:
                s['adv'] = s['r'] - mean
        train = [s for s in samples if abs(s['adv']) > 1e-6]  # all-equal groups carry no signal

        model.train()
        for i in range(0, len(train), args.micro):
            mb = train[i:i + args.micro]
            seqs = [s['p'] + s['c'] for s in mb]
            n, kc = max(map(len, seqs)), max(len(s['c']) for s in mb)
            ids = torch.tensor([[tok.pad_token_id] * (n - len(x)) + x for x in seqs], device=dev)
            mask = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in seqs], device=dev)
            with torch.autocast(dev, dtype=dtype, enabled=dev == 'cuda'):
                logits = model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                               logits_to_keep=kc + 1, use_cache=False).logits
            lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, -kc:].unsqueeze(-1)).squeeze(-1)
            cm = torch.tensor([[0] * (kc - len(s['c'])) + [1] * len(s['c']) for s in mb], device=dev)
            adv = torch.tensor([s['adv'] for s in mb], device=dev)
            loss = -(adv * (lp * cm).sum(-1) / cm.sum(-1)).sum() / len(train)
            scaler.scale(loss).backward()
        if train:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        mixed = sum(len({round(s['r'], 6) for s in grp}) > 1 for grp in groups.values())
        print(f"step {step + 1}/{args.steps}  reward {sum(s['r'] for s in samples) / len(samples):.3f}  "
              f"ask {sum(s['asked'] for s in samples) / len(samples):.2f}  groups with signal {mixed}/{len(groups)}  "
              f"trained {len(train)}  len {sum(len(s['c']) for s in samples) / len(samples):.0f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if (step + 1) % args.save_steps == 0 or step + 1 == args.steps:
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
    print(f'done in {(time.time() - t0) / 60:.1f} min; adapter saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
