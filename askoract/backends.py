import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from askoract.prompt import prompt_text


class HF:
    def __init__(self, model, adapter, batch):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.batch = torch, batch
        self.tok = AutoTokenizer.from_pretrained(model)  # adapters keep the base tokenizer
        self.tok.padding_side = 'left'
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map={'': 0})
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.m = m.eval()

    def __call__(self, convs, batch=None):
        outs = []
        b = batch or self.batch
        for i in range(0, len(convs), b):
            texts = [prompt_text(self.tok, c) for c in convs[i:i + b]]
            enc = self.tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.m.device)
            with self.torch.no_grad():
                g = self.m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=self.tok.pad_token_id)
            outs += self.tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        return outs


EXTRA = {'qwen/qwen3.8-flash': {'enable_thinking': False}}  # thinking on takes minutes per call on our endpoint
LOCAL = {'temperature': 0, 'max_tokens': 80, 'chat_template_kwargs': {'enable_thinking': False}}  # our model, greedy


class API:
    def __init__(self, model, workers=8, timeout=240, url=None):
        if url:  # local server (mlx_lm.server / vLLM), no key needed
            self.key, self.url, self.extra = 'none', url.rstrip('/') + '/chat/completions', LOCAL
        else:
            env = dict(l.strip().split('=', 1) for l in open(Path(__file__).resolve().parents[2] / '.env')
                       if '=' in l and not l.startswith('#'))
            self.key = env['LITELLM_API_KEY'].strip().strip('"\'')
            self.url = env['LITELLM_BASE_URL'].strip().strip('"\'').rstrip('/') + '/chat/completions'
            self.extra = EXTRA.get(model, {})
        self.model, self.workers, self.timeout = model, workers, timeout

    def one(self, msgs):
        body = json.dumps({'model': self.model, 'messages': msgs, **self.extra}).encode()
        req = urllib.request.Request(self.url, body, {'Authorization': f'Bearer {self.key}',
                                                      'Content-Type': 'application/json'})
        for attempt in range(3):
            try:
                return json.load(urllib.request.urlopen(req, timeout=self.timeout))['choices'][0]['message']['content'] or ''
            except Exception as ex:  # rate limits, timeouts: retry, then give up
                err = ex
                time.sleep(2 ** attempt)
        return f'ERROR {type(err).__name__}'

    def __call__(self, convs, batch=None):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self.one, convs))
