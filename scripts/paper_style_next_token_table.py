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
from probing import prompting  # noqa: E402

POSITIVE_WORDS = {
    "neat", "cooperative", "coherent", "logical",
    "calm", "oriented", "attentive", "insightful",
}
NEGATIVE_WORDS = {
    "unkempt", "guarded", "restless", "slowed", "tangential", "paranoid",
    "depressed", "anxious", "irritable", "confused", "distracted", "impulsive",
}


def safe_name(model: str) -> str:
    return model.replace('/', '__').replace(':', '__')


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
            x = item_pat.sub("", line).strip()
            if x:
                current.append(x)
        else:
            x = line.strip()
            if x:
                current.append(x)
    if current:
        blocks.append(" ".join(current).strip())
    return blocks


def load_pairs_from_file(path: Path, n: int):
    lines = [x.strip() for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    pairs = []
    for ln in lines[:n]:
        a, b = ln.split('\t')
        pairs.append((a, b))
    return pairs


def load_sae_as_pairs(top500_path: Path, n: int):
    blocks = parse_numbered_blocks(top500_path.read_text(encoding='utf8'))
    blocks = blocks[:n]
    return [(x, x) for x in blocks]


def load_attributes(path: Path, tok):
    attrs = [x.strip() for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    kept_words, kept_ids, skipped = [], [], []
    for a in attrs:
        toks = tok.tokenize(' ' + a)
        if len(toks) == 1:
            kept_words.append(a)
            kept_ids.append(tok.convert_tokens_to_ids(toks[0]))
        else:
            skipped.append(a)
    if skipped:
        print('[single-token filter] skipped:', ', '.join(skipped))
    if len(kept_words) < 2:
        raise ValueError('Need >=2 single-token adjectives for paper-style next-token probing.')
    return kept_words, kept_ids


def load_attributes_sequence(path: Path):
    attrs = [x.strip() for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    if len(attrs) < 2:
        raise ValueError('Need >=2 adjectives for sequence probing.')
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
            cont_ids = tok.encode(' ' + adj, add_special_tokens=False)
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
    ap.add_argument('--model', required=True, help='e.g., gpt2-medium or mistralai/Mistral-7B-Instruct-v0.3')
    ap.add_argument('--attributes', default='data/attributes/mental_attitudes.txt')
    ap.add_argument('--pair_file', default=None, help='TSV with sae<TAB>aae; if omitted, use top500 as SAE=AAE baseline')
    ap.add_argument('--top500', default='top500.txt')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--calibrate', action='store_true', help='Subtract neutral prompt probs (paper-style calibration)')
    ap.add_argument('--scoring', choices=['one_token', 'sequence'], default='one_token')
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--out_md', default=None)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
    model.eval()

    max_len = getattr(model.config, 'max_position_embeddings', None)
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = 1024

    if args.pair_file:
        pairs = load_pairs_from_file(Path(args.pair_file), args.n)
    else:
        pairs = load_sae_as_pairs(Path(args.top500), args.n)

    if args.scoring == 'one_token':
        adjectives, token_ids = load_attributes(Path(args.attributes), tok)
    else:
        adjectives = load_attributes_sequence(Path(args.attributes))
        token_ids = None
    prompts = prompting.MENTAL_HEALTH_ATTITUDE_PROMPTS

    cal = {}
    if args.calibrate:
        for p in prompts:
            if args.scoring == 'one_token':
                cal[p] = next_token_probs(model, tok, p.format(''), token_ids, device, max_len)
            else:
                cal[p] = sequence_scores(model, tok, p.format(''), adjectives, device, max_len)

    sae_rows, aae_rows = [], []
    for p in prompts:
        for sae_text, aae_text in pairs:
            if args.scoring == 'one_token':
                sae_probs = next_token_probs(model, tok, p.format(sae_text), token_ids, device, max_len)
                aae_probs = next_token_probs(model, tok, p.format(aae_text), token_ids, device, max_len)
            else:
                sae_probs = sequence_scores(model, tok, p.format(sae_text), adjectives, device, max_len)
                aae_probs = sequence_scores(model, tok, p.format(aae_text), adjectives, device, max_len)
            if args.calibrate:
                sae_probs = sae_probs - cal[p]
                aae_probs = aae_probs - cal[p]
            sae_rows.append(sae_probs)
            aae_rows.append(aae_probs)

    SAE = np.vstack(sae_rows)
    AAE = np.vstack(aae_rows)
    sae_mean = SAE.mean(axis=0)
    aae_mean = AAE.mean(axis=0)
    delta = aae_mean - sae_mean

    df = pd.DataFrame({
        'adjective': adjectives,
        'sae_mean': sae_mean,
        'aae_mean': aae_mean,
        'delta_aae_minus_sae': delta,
        'abs_delta': np.abs(delta),
    }).sort_values('abs_delta', ascending=False)

    stem = f"results/{safe_name(args.model)}_paper_style_{args.scoring}_n{len(pairs)}"
    out_csv = Path(args.out_csv or (stem + '.csv'))
    out_md = Path(args.out_md or (stem + '.md'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    md_df = df.copy()
    md_df['adjective'] = md_df['adjective'].map(tag_word)
    out_md.write_text(md_df.to_markdown(index=False, floatfmt='.6f') + '\n', encoding='utf8')

    print(f'Saved CSV: {out_csv}')
    print(f'Saved MD : {out_md}')
    print(md_df.to_markdown(index=False, floatfmt='.6f'))


if __name__ == '__main__':
    main()
