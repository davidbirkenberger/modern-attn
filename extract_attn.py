# Comments are in English. Runs on CPU or Apple MPS.
import argparse
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from utils_io import read_jsonl_or_json, write_pickle


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def add_special_tokens_if_needed(tokens: List[str], tokenizer) -> List[str]:
    """Ensure [CLS] and [SEP] are present for 'tokens' input mode."""
    tok = list(tokens)
    if tokenizer.cls_token and (not tok or tok[0] != tokenizer.cls_token):
        tok = [tokenizer.cls_token] + tok
    if tokenizer.sep_token and (not tok or tok[-1] != tokenizer.sep_token):
        tok = tok + [tokenizer.sep_token]
    return tok


def encode_example(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """Encode one example supporting 'text' | 'words' | 'tokens'."""
    if "text" in example:
        enc = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    elif "words" in example:
        enc = tokenizer(
            example["words"],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    elif "tokens" in example:
        toks = add_special_tokens_if_needed(example["tokens"], tokenizer)
        ids = tokenizer.convert_tokens_to_ids(toks)
        attn_mask = [1] * len(ids)
        enc = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn_mask], dtype=torch.long),
        }
    else:
        raise ValueError("Each example must have exactly one of: 'text', 'words', 'tokens'.")
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF model id, e.g., bert-base-uncased")
    ap.add_argument("--input-json", required=True, help="Path to JSON/JSONL list of examples")
    ap.add_argument("--output-pkl", required=True, help="Where to write the pickle dump")
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, output_attentions=True).to(device).eval()

    data = read_jsonl_or_json(args.input_json)
    out = []

    for ex in tqdm(data, desc="Extracting attentions"):
        enc = encode_example(ex, tokenizer, args.max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            res = model(**enc)
        attn = torch.stack([a[0] for a in res.attentions]).to("cpu").to(torch.float32).numpy()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].to("cpu").tolist())

        ex_out = dict(ex)
        if "text" not in ex_out:
            if "words" in ex_out:
                ex_out["text"] = " ".join(ex_out["words"])
            else:
                ex_out["text"] = tokenizer.convert_tokens_to_string(tokens)

        ex_out["tokens"] = tokens if "tokens" not in ex_out else ex_out["tokens"]
        ex_out["attns"] = attn
        out.append(ex_out)

    write_pickle(out, args.output_pkl)
    print(f"Wrote {len(out)} examples to {args.output_pkl}")


if __name__ == "__main__":
    main()