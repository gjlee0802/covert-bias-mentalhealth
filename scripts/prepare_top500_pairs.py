#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from prepare_mentalchat_pairs import convert_sae_to_aae, normalize


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
            stripped = line.strip()
            if stripped:
                current.append(stripped)

    if current:
        blocks.append(" ".join(current).strip())

    return blocks


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="top500.txt")
    parser.add_argument("--out_dir", default="data/pairs")
    parser.add_argument("--prefix", default="top500")
    parser.add_argument("--max_rows", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf8")
    blocks = parse_numbered_blocks(text)

    pairs_sae_aae = []
    for b in blocks:
        sae = normalize(b)
        aae = normalize(convert_sae_to_aae(sae))
        pairs_sae_aae.append(f"{sae}\t{aae}")
        if len(pairs_sae_aae) >= args.max_rows:
            break

    if not pairs_sae_aae:
        raise ValueError("No blocks parsed from top500 input.")

    pairs_sae_sae = [f"{x.split(chr(9))[0]}\t{x.split(chr(9))[0]}" for x in pairs_sae_aae]

    out_dir = Path(args.out_dir)
    p1 = out_dir / f"{args.prefix}_sae_aae.txt"
    p2 = out_dir / f"{args.prefix}_sae_sae.txt"

    write_lines(p1, pairs_sae_aae)
    write_lines(p2, pairs_sae_sae)

    print(f"Saved {len(pairs_sae_aae)} pairs to {p1}")
    print(f"Saved {len(pairs_sae_sae)} pairs to {p2}")


if __name__ == "__main__":
    main()
