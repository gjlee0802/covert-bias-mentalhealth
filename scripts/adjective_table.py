#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def load(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def aggregate(obj):
    attrs = None
    sae_rows, aae_rows = [], []
    for _, rows in obj.items():
        for _, cls, attributes, probs in rows:
            attrs = attributes
            arr = np.array(probs, dtype=float)
            if cls == 'sae':
                sae_rows.append(arr)
            elif cls == 'aave':
                aae_rows.append(arr)
    sae = np.vstack(sae_rows)
    aae = np.vstack(aae_rows)
    sae_m = sae.mean(axis=0)
    aae_m = aae.mean(axis=0)
    delta = aae_m - sae_m
    out = pd.DataFrame({
        'adjective': attrs,
        'mean_sae': sae_m,
        'mean_aae': aae_m,
        'delta_aae_minus_sae': delta,
        'abs_delta': np.abs(delta),
    }).sort_values('abs_delta', ascending=False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='e.g., hf:Qwen/Qwen3-8B')
    p.add_argument('--prefix', default='top500')
    p.add_argument('--num_pairs', type=int, default=500)
    p.add_argument('--calibrated', action='store_true')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--out_csv', default=None)
    args = p.parse_args()

    safe_model = args.model.replace('/', '__').replace(':', '__')
    n_tag = f'_n{args.num_pairs}'
    suffix = '_cal.p' if args.calibrated else '.p'
    f = Path(args.results_dir) / f'{safe_model}_{args.prefix}_sae_aae_mental_attitudes{n_tag}{suffix}'

    if not f.exists():
        raise FileNotFoundError(f'Missing results file: {f}')

    df = aggregate(load(f))
    print(df.to_markdown(index=False, floatfmt='.6f'))

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f'\nSaved CSV: {args.out_csv}')


if __name__ == '__main__':
    main()
