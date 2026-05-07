#!/usr/bin/env python3
import argparse
import pickle
import random

import numpy as np
import torch
import tqdm

import helpers


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--attribute", type=str, default="mental_attitudes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=123)
    args = parser.parse_args()

    model = helpers.load_model(args.model)
    tok = helpers.load_tokenizer(args.model)

    prompts, cal_prompts = helpers.load_prompts(args.model)
    variable_classes = ["sae", "aave"]
    attribute_classes = helpers.load_attributes(args.attribute, tok, args.model)
    variable_pairs = helpers.load_pairs(args.variable)
    if args.max_pairs is not None and args.max_pairs < len(variable_pairs):
        rng = random.Random(args.sample_seed)
        variable_pairs = rng.sample(variable_pairs, args.max_pairs)
        print(f"Sampled {len(variable_pairs)} pairs (seed={args.sample_seed})")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if not helpers.is_openai_model(args.model):
        model = model.to(device)

    labels = None
    if helpers.normalize_model_name(args.model) in helpers.T5_MODELS:
        labels = torch.tensor([tok.encode("<extra_id_0>")]).to(device)

    prompt_cal_probs = {}
    if args.calibrate:
        if helpers.is_openai_model(args.model):
            for prompt, cal_prompt in zip(prompts, cal_prompts):
                prompt_cal_probs[prompt] = helpers.get_attribute_probs(
                    cal_prompt,
                    attribute_classes,
                    model,
                    args.model,
                    tok,
                    device,
                    labels,
                )
        else:
            model.eval()
            with torch.no_grad():
                for prompt, cal_prompt in zip(prompts, cal_prompts):
                    prompt_cal_probs[prompt] = helpers.get_attribute_probs(
                        cal_prompt,
                        attribute_classes,
                        model,
                        args.model,
                        tok,
                        device,
                        labels,
                    )

    prompt_results = {}
    if helpers.is_openai_model(args.model):
        for prompt in prompts:
            results = []
            for variable_pair in tqdm.tqdm(variable_pairs):
                sae, aae = variable_pair.strip().split("\t")
                for i, variable in enumerate([sae, aae]):
                    probs = helpers.get_attribute_probs(
                        prompt.format(variable),
                        attribute_classes,
                        model,
                        args.model,
                        tok,
                        device,
                        labels,
                    )
                    if args.calibrate:
                        probs = helpers.calibrate(probs, prompt_cal_probs[prompt])
                    results.append((
                        variable,
                        variable_classes[i],
                        attribute_classes,
                        probs,
                    ))
            prompt_results[prompt] = results
    else:
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                results = []
                for variable_pair in tqdm.tqdm(variable_pairs):
                    sae, aae = variable_pair.strip().split("\t")
                    for i, variable in enumerate([sae, aae]):
                        probs = helpers.get_attribute_probs(
                            prompt.format(variable),
                            attribute_classes,
                            model,
                            args.model,
                            tok,
                            device,
                            labels,
                        )
                        if args.calibrate:
                            probs = helpers.calibrate(probs, prompt_cal_probs[prompt])
                        results.append((
                            variable,
                            variable_classes[i],
                            attribute_classes,
                            probs,
                        ))
                prompt_results[prompt] = results

    suffix = "_cal.p" if args.calibrate else ".p"
    safe_model = args.model.replace("/", "__").replace(":", "__")
    n_tag = f"_n{len(variable_pairs)}"
    output = f"{helpers.RESULTS_PATH}/{safe_model}_{args.variable}_{args.attribute}{n_tag}{suffix}"
    with open(output, "wb") as f:
        pickle.dump(prompt_results, f)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
