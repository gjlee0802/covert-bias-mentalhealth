#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TEXT_KEYS = [
    "text",
    "utterance",
    "input",
    "prompt",
    "query",
    "instruction",
    "user",
    "question",
    "message",
]


AAE_RULES = [
    (r"\bisn['’]?t\b", "ain't"),
    (r"\baren['’]?t\b", "ain't"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "don't"),
    (r"\bdid not\b", "didn't"),
    (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"),
    (r"\bgoing to\b", "gon'"),
    (r"\bwant to\b", "wanna"),
    (r"\bkind of\b", "kinda"),
    (r"\bsort of\b", "sorta"),
    (r"\bthem\b", "'em"),
    (r"\bmy\b", "mah"),
    (r"\byour\b", "yo"),
    (r"\byou all\b", "y'all"),
    (r"\bfor real\b", "fo real"),
]


def load_records(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["data", "train", "records", "examples"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
        raise ValueError("Unsupported JSON schema. Expected list-like records.")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf8") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    if suffix == ".txt":
        with path.open("r", encoding="utf8") as f:
            return [{"text": line.strip()} for line in f if line.strip()]
    raise ValueError(f"Unsupported file format: {suffix}")


def extract_text(record: Dict) -> Optional[str]:
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ["conversation", "dialogue", "messages", "chat"]:
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    role = str(item.get("role", "")).lower()
                    content = item.get("content") or item.get("text") or item.get("utterance")
                    if isinstance(content, str) and content.strip() and role in {"user", "patient", "client", "human", ""}:
                        return content.strip()
    return None


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if len(src) > 1 and src[0].isupper() and src[1:].islower():
        return repl[0].upper() + repl[1:]
    return repl


def convert_sae_to_aae(text: str) -> str:
    out = text
    for pattern, replacement in AAE_RULES:
        out = re.sub(
            pattern,
            lambda m: apply_case(m.group(0), replacement),
            out,
            flags=re.IGNORECASE,
        )

    # Light phonological stylization
    out = re.sub(r"\b([A-Za-z]+)ing\b", lambda m: m.group(1) + "in'", out)
    out = re.sub(r"\babout\b", "'bout", out, flags=re.IGNORECASE)
    out = re.sub(r"\bwith\b", "wit", out, flags=re.IGNORECASE)
    return out


def build_pairs(records: Iterable[Dict], max_rows: int, min_chars: int) -> List[str]:
    pairs = []
    for record in records:
        text = extract_text(record)
        if not text:
            continue
        sae = normalize(text)
        if len(sae) < min_chars:
            continue
        aae = normalize(convert_sae_to_aae(sae))
        if sae == aae:
            continue
        pairs.append(f"{sae}\t{aae}")
        if len(pairs) >= max_rows:
            break
    return pairs


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to MentalChat16K file (json/jsonl/csv/tsv/txt)")
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "pairs"),
        help="Output directory for pair files",
    )
    parser.add_argument("--max_rows", type=int, default=16000, help="Maximum number of pairs")
    parser.add_argument("--min_chars", type=int, default=10, help="Minimum utterance length")
    parser.add_argument("--prefix", default="mentalchat16k", help="Output file prefix")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    pairs_sae_aae = build_pairs(records, args.max_rows, args.min_chars)
    if not pairs_sae_aae:
        raise ValueError("No usable examples found. Check input schema or lower --min_chars.")

    pairs_sae_sae = [f"{line.split(chr(9))[0]}\t{line.split(chr(9))[0]}" for line in pairs_sae_aae]

    out_dir = Path(args.out_dir)
    path_sae_aae = out_dir / f"{args.prefix}_sae_aae.txt"
    path_sae_sae = out_dir / f"{args.prefix}_sae_sae.txt"

    write_lines(path_sae_aae, pairs_sae_aae)
    write_lines(path_sae_sae, pairs_sae_sae)

    print(f"Saved {len(pairs_sae_aae)} pairs to: {path_sae_aae}")
    print(f"Saved {len(pairs_sae_sae)} pairs to: {path_sae_sae}")


if __name__ == "__main__":
    main()
