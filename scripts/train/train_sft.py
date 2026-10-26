import argparse
import json
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from askoract.prompt import prompt_text


def encode(tok, row, max_len):
    msgs = row['messages']
    p = tok(prompt_text(tok, msgs[:-1]), add_special_tokens=False)['input_ids']
    c = tok(msgs[-1]['content'] + tok.eos_token, add_special_tokens=False)['input_ids']
    ids = (p + c)[:max_len]
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': ([-100] * len(p) + c)[:max_len]}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--data', default='data/sft/train.jsonl')
    a.add_argument('--model', default='Qwen/Qwen3-1.7B')
    a.add_argument('--out', default='runs/sft-1.7b')
    a.add_argument('--limit', type=int, default=0, help='use only the first N rows (0 = all)')
    a.add_argument('--epochs', type=float, default=1.0)
    a.add_argument('--lr', type=float, default=2e-4)
    a.add_argument('--batch', type=int, default=4)
    a.add_argument('--accum', type=int, default=4)
    a.add_argument('--rank', type=int, default=16)
    a.add_argument('--max-len', type=int, default=1024)
    a.add_argument('--save-steps', type=int, default=100, help='checkpoint interval (free Colab sessions can vanish)')
    a.add_argument('--resume', default=None, help='checkpoint folder to resume from')
    a.add_argument('--init', default=None, help='adapter to continue training (default: new LoRA)')
    a.add_argument('--seed', type=int, default=42, help='LoRA init and data order')
    args = a.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)]
    rows = rows[:args.limit] if args.limit else rows
    data = [encode(tok, r, args.max_len) for r in rows]
    lens = sorted(len(d['input_ids']) for d in data)
    print(f'{len(data)} rows; tokens p50 {lens[len(lens) // 2]}, max {lens[-1]}; eos {tok.eos_token!r}', flush=True)

    bf16 = torch.cuda.get_device_capability()[0] >= 8  # T4 reports bf16 but only emulates it
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float16,
                                                 attn_implementation='sdpa', device_map={'': 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.init, is_trainable=True) if args.init else get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    for p in model.parameters():  # fp16 mixed precision needs fp32 trainable weights
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=args.out, per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
            learning_rate=args.lr, num_train_epochs=args.epochs, lr_scheduler_type='cosine', warmup_steps=0.03,
            logging_steps=10, save_strategy='steps', save_steps=args.save_steps, save_total_limit=1,
            report_to='none', fp16=not bf16, bf16=bf16, seed=args.seed,
            remove_unused_columns=False, dataloader_num_workers=2),
    )
    t = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f'done in {(time.time() - t) / 60:.1f} min; adapter saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
