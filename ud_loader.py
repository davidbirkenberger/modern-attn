# Comments are in English.
# Lightweight UD CoNLL-U loader that returns sentences as dicts with
#   - words:  list[str]
#   - upos:   list[str]
#   - head:   list[int]      (0-based indices; ROOT mapped to self by default)
#   - deprel: list[str]
#   - text:   str            (reconstructed using MISC SpaceAfter=No)
#
# You can either import load_conllu() from other scripts or run this file as a
# CLI to dump a JSON list that can be fed into extract_attn.py directly. Any
# extra fields (upos/head/deprel) will be preserved by our extractor.

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def _iter_sentences(path: str) -> Iterator[List[str]]:
    """Yield raw sentence blocks (list of non-empty, non-comment lines)."""
    buf: List[str] = []
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                if buf:
                    yield buf
                    buf = []
                continue
            buf.append(line)
        if buf:
            yield buf


def _parse_misc(misc: str) -> Dict[str, str]:
    """Parse the MISC column (key=value|key=value)."""
    out: Dict[str, str] = {}
    if misc == "_" or not misc:
        return out
    for part in misc.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out[part] = "True"
    return out


def _reconstruct_text(forms: List[str], no_space_after: List[bool]) -> str:
    """Rebuild sentence text using SpaceAfter=No info from MISC.
    If MISC lacked SpaceAfter, we assume a space after the token.
    """
    pieces: List[str] = []
    for i, w in enumerate(forms):
        pieces.append(w)
        if i < len(forms) - 1 and not no_space_after[i]:
            pieces.append(" ")
    return "".join(pieces)


def load_conllu(path: str, map_root_to_self: bool = True) -> List[Dict[str, object]]:
    """Load a UD CoNLL-U file into sentence dicts.

    Args:
        path: Path to *.conllu file
        map_root_to_self: If True, set HEAD of ROOT tokens to their own index (non-negative),
                          else set to -1.
    Returns:
        List of sentence dicts with words/upos/head/deprel/text.
    """
    sentences: List[Dict[str, object]] = []
    for block in _iter_sentences(path):
        words: List[str] = []
        upos: List[str] = []
        head: List[int] = []
        deprel: List[str] = []
        no_space_after: List[bool] = []

        for line in block:
            # Columns: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            cols = line.split("	")
            if len(cols) < 10:
                # Malformed line; skip
                continue
            id_col = cols[0]
            # Skip multiword tokens (e.g., 1-2) and empty nodes (e.g., 3.1)
            if "-" in id_col or "." in id_col:
                continue
            try:
                idx0 = int(id_col) - 1
            except ValueError:
                continue
            form = cols[1]
            up = cols[3]
            head_raw = cols[6]
            dep = cols[7]
            misc = _parse_misc(cols[9])

            # Append core fields
            words.append(form)
            upos.append(up)

            # Convert head to 0-based; handle ROOT (0)
            try:
                h = int(head_raw) - 1
            except ValueError:
                h = -1
            if h < 0:
                h = idx0 if map_root_to_self else -1
            head.append(h)
            deprel.append(dep)
            no_space_after.append(misc.get("SpaceAfter", "Yes") == "No")

        if not words:
            continue
        text = _reconstruct_text(words, no_space_after)
        sentences.append({
            "words": words,
            "upos": upos,
            "head": head,
            "deprel": deprel,
            "text": text,
        })
    return sentences


def dump_json(input_conllu: str, output_json: str, limit: int | None = None) -> None:
    """Read CONLL-U and write a JSON list where each item contains
    words/upos/head/deprel/text. This can be fed to extract_attn.py.
    """
    sents = load_conllu(input_conllu)
    if limit is not None:
        sents = sents[:limit]
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sents, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(sents)} sentences → {output_json}")


def main():
    ap = argparse.ArgumentParser(description="UD CoNLL-U → JSON loader for attention analysis")
    ap.add_argument("--input", required=True, help="Path to *.conllu (e.g., UD_German-GSD train/dev/test)")
    ap.add_argument("--output", required=True, help="Path to output JSON list")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of sentences")
    ap.add_argument("--root-as-self", action="store_true", help="Map ROOT head to self index (default)")
    ap.add_argument("--root-as-neg1", action="store_true", help="Map ROOT head to -1 instead")
    args = ap.parse_args()

    if args.root_as_self and args.root_as_neg1:
        raise SystemExit("Choose at most one of --root-as-self or --root-as-neg1")

    map_root_to_self = not args.root_as_neg1  # default True
    sents = load_conllu(args.input, map_root_to_self=map_root_to_self)
    if args.limit is not None:
        sents = sents[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sents, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(sents)} sentences → {args.output}")


if __name__ == "__main__":
    main()