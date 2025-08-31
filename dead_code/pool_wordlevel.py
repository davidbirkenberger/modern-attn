# Comments are in English. Pools token-level attention to word-level via mean pooling.
import argparse
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer
from utils_io import read_pickle, write_pickle


def build_word_to_subword_from_words(tokenizer, words):
    """Map UD words to subword indices using fast tokenizer's word_ids()."""
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    w2s = [[] for _ in range(len(words))]
    for si, wid in enumerate(word_ids):
        if wid is None:  # special tokens like [CLS]/[SEP]
            continue
        w2s[wid].append(si)
    return w2s

def pool_matrix_wordlevel_sum(A_TT: np.ndarray, w2s):
    """Sum-pool token-level attention (T,T) to word-level (W,W), then row-normalize."""
    W, T = len(w2s), A_TT.shape[0]
    # Row pooling: sum source subpieces
    rowpool = np.zeros((W, T), dtype=np.float32)
    for wi, idxs in enumerate(w2s):
        if idxs:
            rowpool[wi] = A_TT[idxs, :].sum(axis=0)
    # Column pooling: sum target subpieces
    A_WW = np.zeros((W, W), dtype=np.float32)
    for wj, idxs in enumerate(w2s):
        if idxs:
            A_WW[:, wj] = rowpool[:, idxs].sum(axis=1)
    # Row-normalize for readability
    rs = A_WW.sum(axis=1, keepdims=True)
    A_WW = np.divide(A_WW, np.maximum(rs, 1e-9))
    return A_WW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--input-pkl", required=True)
    ap.add_argument("--output-pkl", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    data = read_pickle(args.input_pkl)
    out = []

    for ex in tqdm(data, desc="Pooling to word level"):
        attn = ex["attns"]  # (L,H,T,T)
        # Prefer gold UD words if available
        if "words" in ex and ex["words"]:
            words = ex["words"]
            w2s = build_word_to_subword_from_words(tok, words)
        else:
            # Fallback: reconstruct words via whitespace + offsets (your original logic)
            w2s, words = build_word_to_subword(tok, ex.get("text", ""))

        L, H = attn.shape[:2]
        pooled = np.zeros((L, H, len(w2s), len(w2s)), dtype=attn.dtype)
        for l in range(L):
            for h in range(H):
                pooled[l, h] = pool_matrix_wordlevel_sum(attn[l, h], w2s)

        ex2 = dict(ex)
        ex2["attns"] = pooled           # now (L,H,W,W)
        ex2["words"] = words            # UD words (or fallback)
        ex2["pooled_level"] = "word"    # handy flag for plotting
        out.append(ex2)

    write_pickle(out, args.output_pkl)
    print(f"Wrote {len(out)} examples to {args.output_pkl}")


if __name__ == "__main__":
    main()