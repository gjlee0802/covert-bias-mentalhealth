import os
import time
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    RobertaForMaskedLM,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

import prompting

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import tiktoken
except Exception:
    tiktoken = None

ATTRIBUTES_PATH = os.path.abspath("../data/attributes/{}.txt")
VARIABLES_PATH = os.path.abspath("../data/pairs/{}.txt")
RESULTS_PATH = os.path.abspath("../results")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
ROBERTA_MODELS = ["roberta-base", "roberta-large"]
T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b"]


def is_openai_model(model_name: str) -> bool:
    return model_name.startswith("openai:")


def is_hf_prefixed(model_name: str) -> bool:
    return model_name.startswith("hf:")


def normalize_model_name(model_name: str) -> str:
    if is_openai_model(model_name):
        return model_name.split(":", 1)[1]
    if is_hf_prefixed(model_name):
        return model_name.split(":", 1)[1]
    return model_name


def is_sequence_scoring_model(model_name: str) -> bool:
    raw = normalize_model_name(model_name)
    return is_openai_model(model_name) or is_hf_prefixed(model_name) or raw in GPT2_MODELS


def load_model(model_name):
    raw_name = normalize_model_name(model_name)

    if is_openai_model(model_name):
        if OpenAI is None:
            raise ImportError("openai package is required for openai:* models.")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key, base_url=base_url)

    if raw_name in GPT2_MODELS:
        return GPT2LMHeadModel.from_pretrained(raw_name)
    if raw_name in ROBERTA_MODELS:
        return RobertaForMaskedLM.from_pretrained(raw_name)
    if raw_name in T5_MODELS:
        return T5ForConditionalGeneration.from_pretrained(raw_name)

    return AutoModelForCausalLM.from_pretrained(raw_name, trust_remote_code=True)


def load_tokenizer(model_name):
    raw_name = normalize_model_name(model_name)

    if is_openai_model(model_name):
        return None

    if raw_name in GPT2_MODELS:
        return GPT2Tokenizer.from_pretrained(raw_name)
    if raw_name in ROBERTA_MODELS:
        return RobertaTokenizer.from_pretrained(raw_name)
    if raw_name in T5_MODELS:
        return T5Tokenizer.from_pretrained(raw_name)

    tok = AutoTokenizer.from_pretrained(raw_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_prompts(model_name):
    raw_name = normalize_model_name(model_name)
    prompts = prompting.MENTAL_HEALTH_ATTITUDE_PROMPTS

    if raw_name in ROBERTA_MODELS:
        prompts = [p + " <mask>" for p in prompts]
    elif raw_name in T5_MODELS:
        prompts = [p + " <extra_id_0>" for p in prompts]

    cal_prompts = [p.format("") for p in prompts]
    return prompts, cal_prompts


def load_attributes(attribute_name, tok, model_name):
    with open(ATTRIBUTES_PATH.format(attribute_name), "r", encoding="utf8") as f:
        attributes = f.read().strip().split("\n")

    if is_sequence_scoring_model(model_name):
        return attributes

    for a in attributes:
        assert len(tok.tokenize(" " + a)) == 1, f"Attribute must be single token: {a}"
    return [tok.tokenize(" " + a)[0] for a in attributes]


def load_pairs(variable):
    with open(VARIABLES_PATH.format(variable), "r", encoding="utf8") as f:
        return f.read().strip().split("\n")


def _model_max_positions(model, tok) -> int:
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(max_pos, int) and max_pos > 0:
        return max_pos
    tok_max = getattr(tok, "model_max_length", None)
    if isinstance(tok_max, int) and tok_max > 0 and tok_max < 10**8:
        return tok_max
    return 2048


def _encode_with_truncation(prompt: str, tok, model, model_name: str) -> torch.Tensor:
    raw_name = normalize_model_name(model_name)
    max_len = _model_max_positions(model, tok)

    encoded = tok.encode(prompt, add_special_tokens=False)
    if len(encoded) > max_len:
        encoded = encoded[-max_len:]

    if raw_name in ROBERTA_MODELS:
        encoded = tok.build_inputs_with_special_tokens(encoded)
        if len(encoded) > max_len:
            encoded = encoded[-max_len:]

    return torch.tensor([encoded])


def compute_probs(model, model_name, input_ids, labels):
    raw_name = normalize_model_name(model_name)

    if raw_name in GPT2_MODELS or is_hf_prefixed(model_name) or (
        raw_name not in ROBERTA_MODELS + T5_MODELS
    ):
        output = model(input_ids=input_ids)
        return F.softmax(output.logits, dim=-1)[0][-1]
    if raw_name in ROBERTA_MODELS:
        output = model(input_ids=input_ids)
        return F.softmax(output.logits, dim=-1)[0][-2]
    if raw_name in T5_MODELS:
        output = model(input_ids=input_ids, labels=labels)
        return F.softmax(output.logits, dim=-1)[0][-1]
    raise ValueError(f"Model {model_name} not supported.")


def _normalize_scores_to_probs(scores: List[float]) -> List[float]:
    arr = np.array(scores, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = np.sum(exp)
    if denom <= 0:
        return [0.0 for _ in scores]
    return (exp / denom).tolist()


def _openai_single_token_fallback_probs(prompt: str, attrs: List[str], client, openai_model: str) -> List[float]:
    if tiktoken is None:
        raise ImportError("tiktoken package is required for openai:* models.")
    try:
        enc = tiktoken.encoding_for_model(openai_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    attr_token_ids = {}
    skipped = []
    for a in attrs:
        ids = enc.encode(" " + a)
        if len(ids) == 1:
            attr_token_ids[a] = ids[0]
        else:
            skipped.append(a)
    if skipped:
        print(f"[openai fallback] multi-token attributes set to 0 prob: {', '.join(skipped)}")
    if len(attr_token_ids) < 2:
        raise RuntimeError("OpenAI fallback needs at least 2 single-token attributes.")

    logit_bias = {str(tid): 100 for tid in attr_token_ids.values()}
    last_err = None
    top = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16,
                temperature=0,
                logprobs=True,
                top_logprobs=min(len(attr_token_ids), 20),
                logit_bias=logit_bias,
            )
            lp = getattr(response.choices[0], "logprobs", None)
            content = getattr(lp, "content", None) if lp is not None else None
            if content and len(content) > 0 and getattr(content[0], "top_logprobs", None):
                top = content[0].top_logprobs
                break
            last_err = RuntimeError("Provider response did not include logprobs.")
        except Exception as e:
            last_err = e
        time.sleep(0.6 * (attempt + 1))

    if top is None:
        raise RuntimeError(f"OpenAI fallback failed. Model={openai_model}. Last error={last_err}")

    token_to_logprob = {entry.token: entry.logprob for entry in top}
    probs = []
    for a in attrs:
        tok = " " + a
        logp = token_to_logprob.get(tok, -9999.0)
        probs.append(np.exp(logp) if logp > -1000 else 0.0)
    return probs


def _openai_sequence_probs(prompt: str, attrs: List[str], client, openai_model: str) -> List[float]:
    base_url = str(getattr(client, "base_url", ""))
    # OpenRouter generally expects provider-prefixed IDs, e.g., openai/gpt-4.1-mini.
    if "openrouter.ai" in base_url and "/" not in openai_model:
        openai_model = f"openai/{openai_model}"

    if tiktoken is None:
        raise ImportError("tiktoken package is required for openai:* models.")
    try:
        enc = tiktoken.encoding_for_model(openai_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    scores = []
    for attr in attrs:
        continuation = " " + attr
        prompt_len = len(enc.encode(prompt))
        cont_len = len(enc.encode(continuation))
        if cont_len == 0:
            scores.append(-1e9)
            continue

        token_logprobs = None
        last_err = None
        for attempt in range(4):
            try:
                response = client.completions.create(
                    model=openai_model,
                    prompt=prompt + continuation,
                    max_tokens=16,
                    temperature=0,
                    logprobs=1,
                    echo=True,
                )
                lp = getattr(response.choices[0], "logprobs", None)
                token_logprobs = getattr(lp, "token_logprobs", None) if lp is not None else None
                if token_logprobs:
                    break
                last_err = RuntimeError("Provider response did not include token_logprobs.")
            except Exception as e:
                last_err = e
            time.sleep(0.6 * (attempt + 1))

        if not token_logprobs:
            raise RuntimeError(
                f"OpenAI sequence scoring failed for attribute '{attr}'. "
                f"Model={openai_model}. Last error={last_err}"
            )

        # Use continuation token positions from the echoed prompt segment only.
        start = prompt_len
        end = prompt_len + cont_len
        tail = token_logprobs[start:end]
        if len(tail) != cont_len or any(x is None for x in tail):
            scores.append(-1e9)
            continue

        # length-normalized sequence score
        scores.append(float(np.mean(tail)))

    return _normalize_scores_to_probs(scores)


def _hf_sequence_probs(prompt: str, attrs: List[str], model, tok, device) -> List[float]:
    max_len = _model_max_positions(model, tok)
    scores = []

    for attr in attrs:
        continuation = " " + attr
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        cont_ids = tok.encode(continuation, add_special_tokens=False)
        if len(cont_ids) == 0:
            scores.append(-1e9)
            continue

        room_for_prompt = max_len - len(cont_ids)
        if room_for_prompt < 1:
            scores.append(-1e9)
            continue

        prompt_ids = prompt_ids[-room_for_prompt:]
        ids = prompt_ids + cont_ids
        input_ids = torch.tensor([ids], device=device)

        output = model(input_ids=input_ids)
        log_probs = F.log_softmax(output.logits, dim=-1)[0]

        start = len(prompt_ids)
        token_logps = []
        for pos in range(start, len(ids)):
            prev_pos = pos - 1
            if prev_pos < 0:
                continue
            tok_id = ids[pos]
            token_logps.append(log_probs[prev_pos, tok_id].item())

        if not token_logps:
            scores.append(-1e9)
            continue

        scores.append(float(np.mean(token_logps)))

    return _normalize_scores_to_probs(scores)


def get_attribute_probs(prompt, attributes, model, model_name, tok, device, labels):
    if is_openai_model(model_name):
        resolved_model = normalize_model_name(model_name)
        try:
            return _openai_sequence_probs(prompt, attributes, model, resolved_model)
        except RuntimeError as e:
            msg = str(e)
            if "token_logprobs" in msg:
                print("[openai fallback] token_logprobs unavailable; switching to single-token scoring.")
                return _openai_single_token_fallback_probs(prompt, attributes, model, resolved_model)
            raise

    if is_hf_prefixed(model_name) or normalize_model_name(model_name) in GPT2_MODELS:
        return _hf_sequence_probs(prompt, attributes, model, tok, device)

    input_ids = _encode_with_truncation(prompt, tok, model, model_name).to(device)
    probs = compute_probs(model, model_name, input_ids, labels)
    return [probs[tok.convert_tokens_to_ids(a)].item() for a in attributes]


def calibrate(probs, cal_probs):
    return [(p - cp) for p, cp in zip(probs, cal_probs)]


def mean_attribute_deltas(prompt_results):
    deltas = []
    attrs = None
    for _, rows in prompt_results.items():
        sae = []
        aae = []
        for _, cls, attributes, probs in rows:
            attrs = attributes
            if cls == "sae":
                sae.append(np.array(probs, dtype=float))
            else:
                aae.append(np.array(probs, dtype=float))
        sae_mean = np.vstack(sae).mean(axis=0)
        aae_mean = np.vstack(aae).mean(axis=0)
        deltas.append(aae_mean - sae_mean)
    return attrs, np.vstack(deltas).mean(axis=0)
