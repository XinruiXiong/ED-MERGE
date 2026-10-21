"""
Label HPI sentences (K / P / D) with a local open-weight model; see hpi_clean.py.

Model: Qwen2.5-7B-Instruct (the model used for the reported external
validation). --model takes the local model directory; any chat model that
supports tokenizer.apply_chat_template works.

Greedy decoding, so a rerun with the same model reproduces the labels.

Usage:
  python -m external_validation.hpi_clean_label INPUT.jsonl OUTPUT.jsonl [--model PATH]
Resumable: items already in OUTPUT are skipped.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from external_validation.hpi_clean import SYSTEM_PROMPT, user_prompt, parse_labels, cleaned_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL = "<PLACEHOLDER_QWEN2_5_7B_INSTRUCT_DIR>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.input)]
    done = set()
    if os.path.exists(args.output):
        done = {json.loads(l)["id"] for l in open(args.output)}
    todo = [it for it in items if it["id"] not in done]
    log.info("items %d, done %d, todo %d", len(items), len(done), len(todo))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    def prompts_for(batch):
        return [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt(it["sentences"])}],
            tokenize=False, add_generation_prompt=True) for it in batch]

    todo.sort(key=lambda it: sum(len(s["text"]) for s in it["sentences"]))
    with open(args.output, "a") as f:
        for b in range(0, len(todo), args.batch):
            batch = todo[b:b + args.batch]
            enc = tok(prompts_for(batch), return_tensors="pt", padding=True).to(model.device)
            max_new = 12 * max(len(it["sentences"]) for it in batch) + 60
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for it, raw in zip(batch, outs):
                labels = parse_labels(raw, len(it["sentences"]))
                rec = {k: v for k, v in it.items()}
                rec.update(raw_output=raw, labels=labels,
                           cleaned=cleaned_text(it["sentences"], labels) if labels else None)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            log.info("%d / %d", min(b + args.batch, len(todo)), len(todo))


if __name__ == "__main__":
    main()
