#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from probing import prompting

POSITIVE_WORDS = {
    "neat", "cooperative", "coherent", "logical",
    "calm", "oriented", "attentive", "insightful",
}
NEGATIVE_WORDS = {
    "unkempt", "guarded", "restless", "slowed", "tangential", "paranoid",
    "depressed", "anxious", "irritable", "confused", "distracted", "impulsive",
}


def parse_numbered_blocks(text: str):
    lines = text.splitlines()
    blocks = []
    current = []
    item_pat = re.compile(r"^\s*\d+\.\s+")

    for line in lines:
        if item_pat.match(line):
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            line = item_pat.sub("", line).strip()
            if line:
                current.append(line)
        else:
            s = line.strip()
            if s:
                current.append(s)

    if current:
        blocks.append(" ".join(current).strip())
    return blocks


def load_sae_texts(top500_path: Path, n: int):
    text = top500_path.read_text(encoding="utf8")
    blocks = parse_numbered_blocks(text)
    if len(blocks) < n:
        raise ValueError(f"Requested {n} but only found {len(blocks)} items in {top500_path}")
    return blocks[:n]


def load_attributes(path: Path, tok):
    attrs = [x.strip() for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    kept = []
    skipped = []
    for a in attrs:
        toks = tok.tokenize(" " + a)
        if len(toks) == 1:
            kept.append((a, toks[0]))
        else:
            skipped.append(a)
    if skipped:
        print("[single-token filter] skipped:", ", ".join(skipped))
    if len(kept) < 2:
        raise ValueError("Need at least 2 single-token adjectives for next-token probing.")
    return kept


def load_attributes_sequence(path: Path):
    attrs = [x.strip() for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    if len(attrs) < 2:
        raise ValueError("Need at least 2 adjectives for sequence probing.")
    return attrs


def next_token_probs(model, tok, prompt_text, token_ids, device, max_len):
    ids = tok.encode(prompt_text, add_special_tokens=False)
    if len(ids) > max_len:
        ids = ids[-max_len:]
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
    return np.array([probs[i].item() for i in token_ids], dtype=float)


def sequence_scores(model, tok, prompt_text, adjectives, device, max_len):
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    out = []
    with torch.no_grad():
        for adj in adjectives:
            cont_ids = tok.encode(" " + adj, add_special_tokens=False)
            if len(cont_ids) == 0:
                out.append(-1e9)
                continue
            room = max_len - len(cont_ids)
            if room < 1:
                out.append(-1e9)
                continue
            p_ids = prompt_ids[-room:]
            ids = p_ids + cont_ids
            input_ids = torch.tensor([ids], device=device)
            logits = model(input_ids=input_ids).logits[0]
            log_probs = F.log_softmax(logits, dim=-1)
            start = len(p_ids)
            token_logps = []
            for pos in range(start, len(ids)):
                token_logps.append(log_probs[pos - 1, ids[pos]].item())
            out.append(float(np.mean(token_logps)) if token_logps else -1e9)
    return np.array(out, dtype=float)


def tag_word(word: str) -> str:
    if word in POSITIVE_WORDS:
        return f"{word} (p)"
    if word in NEGATIVE_WORDS:
        return f"{word} (n)"
    return word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--top500", default="top500.txt")
    ap.add_argument("--attributes", default="data/attributes/mental_attitudes.txt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true", help="Subtract neutral prompt probs like paper setup")
    ap.add_argument("--scoring", choices=["one_token", "sequence"], default="one_token")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    max_len = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = 1024

    texts = load_sae_texts(Path(args.top500), args.n)
    if args.scoring == "one_token":
        attr_pairs = load_attributes(Path(args.attributes), tok)
        attr_words = [w for w, _ in attr_pairs]
        attr_token_ids = [tok.convert_tokens_to_ids(t) for _, t in attr_pairs]
    else:
        attr_words = load_attributes_sequence(Path(args.attributes))
        attr_token_ids = None

    prompts = prompting.MENTAL_HEALTH_ATTITUDE_PROMPTS

    prompt_cal = {}
    if args.calibrate:
        for p in prompts:
            neutral = p.format("")
            if args.scoring == "one_token":
                prompt_cal[p] = next_token_probs(model, tok, neutral, attr_token_ids, device, max_len)
            else:
                prompt_cal[p] = sequence_scores(model, tok, neutral, attr_words, device, max_len)

    all_rows = []
    for p in prompts:
        for t in texts:
            q = p.format(t)
            if args.scoring == "one_token":
                probs = next_token_probs(model, tok, q, attr_token_ids, device, max_len)
            else:
                probs = sequence_scores(model, tok, q, attr_words, device, max_len)
            if args.calibrate:
                probs = probs - prompt_cal[p]
            all_rows.append(probs)

    M = np.vstack(all_rows)
    mean = M.mean(axis=0)
    std = M.std(axis=0)

    df = pd.DataFrame({
        "adjective": attr_words,
        "mean_prob": mean,
        "std_prob": std,
        "rank": np.argsort(-mean).argsort() + 1,
    }).sort_values("mean_prob", ascending=False)

    safe_model = args.model.replace("/", "__").replace(":", "__")
    default_stem = f"results/{safe_model}_sae_top{args.n}_{args.scoring}_table"
    out_csv = Path(args.out_csv or f"{default_stem}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_md = Path(args.out_md or f"{default_stem}.md")
    md_df = df.copy()
    md_df["adjective"] = md_df["adjective"].map(tag_word)
    out_md.write_text(md_df.to_markdown(index=False, floatfmt=".6f") + "\n", encoding="utf8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD : {out_md}")
    print(md_df.to_markdown(index=False, floatfmt='.6f'))


if __name__ == "__main__":
    main()
