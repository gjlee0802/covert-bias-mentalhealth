#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np


def load_probs(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def prompt_means(prompt_results):
    # Each entry: (variable_text, variable_class, attribute_classes, probs_attribute)
    per_prompt = {}
    for prompt, rows in prompt_results.items():
        by_cls = {"aave": [], "sae": []}
        for _, cls, attrs, probs in rows:
            by_cls[cls].append(np.array(probs, dtype=float))
        sae = np.vstack(by_cls["sae"]).mean(axis=0)
        aae = np.vstack(by_cls["aave"]).mean(axis=0)
        per_prompt[prompt] = {
            "attrs": attrs,
            "sae": sae,
            "aave": aae,
            "delta": aae - sae,
        }
    return per_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--probs_dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--prefix", default="mentalchat16k")
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--num_pairs", type=int, default=1000)
    args = parser.parse_args()

    suffix = "_cal.p" if args.calibrated else ".p"
    safe_model = args.model.replace("/", "__").replace(":", "__")
    n_tag = f"_n{args.num_pairs}"
    path_baseline = Path(args.probs_dir) / f"{safe_model}_{args.prefix}_sae_sae_mental_attitudes{n_tag}{suffix}"
    path_compare = Path(args.probs_dir) / f"{safe_model}_{args.prefix}_sae_aae_mental_attitudes{n_tag}{suffix}"

    base = prompt_means(load_probs(path_baseline))
    comp = prompt_means(load_probs(path_compare))

    print(f"Model: {args.model}")
    print(f"Baseline: {path_baseline}")
    print(f"Compare:  {path_compare}\n")

    # Aggregate across prompts
    attrs = None
    deltas = []
    for prompt in comp:
        attrs = comp[prompt]["attrs"]
        deltas.append(comp[prompt]["delta"])
    mean_delta = np.vstack(deltas).mean(axis=0)

    order = np.argsort(-np.abs(mean_delta))
    print("AAE - SAE mean probability delta (absolute sorted):")
    for idx in order:
        print(f"{attrs[idx]:>12s}: {mean_delta[idx]:+.6f}")


if __name__ == "__main__":
    main()
