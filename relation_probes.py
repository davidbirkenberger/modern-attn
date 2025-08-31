# relation_probes.py
# Comments are in English.
# Relation-specific head probes over UD-style inputs.

# Input: a pickle produced by extract_attn.py, where each example ideally contains:
#   - attns:  np.ndarray (L, H, T, T)   # token-level attentions
#   - words:  List[str]                  # UD tokens from ud_loader.py
#   - upos:   List[str]                  # UD UPOS tags
#   - head:   List[int]                  # 0-based head indices (ROOT mapped to self or -1)
#   - deprel: List[str]                  # UD dependency labels
#   - text:   str                        # reconstructed sentence
#
# We re-tokenize the 'words' with the same HF tokenizer to recover the mapping
# from subwords to word indices (via fast tokenizer word_ids). We then pool
# token-level attention (T,T) to word-level (W,W), normalize rows, and compute
# per-head metrics for selected relations (e.g., amod, nsubj, obj, det, case).

import argparse
import json
import pickle
import numpy as np
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer


def build_word_to_subword(tokenizer, words: List[str]) -> List[List[int]]:
    """Return indices of subword tokens belonging to each word using HF fast tokenizer."""
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    w2s: List[List[int]] = [[] for _ in range(len(words))]
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        w2s[wid].append(idx)
    return w2s


def pool_TT_to_WW(A: np.ndarray, w2s: List[List[int]], normalize_rows: bool = True) -> np.ndarray:
    """Pool token-level attention A (T,T) to word-level (W,W) via sum-pooling."""
    W = len(w2s)
    rowpool = np.zeros((W, A.shape[1]), dtype=np.float32)
    for wi, idxs in enumerate(w2s):
        if idxs:
            rowpool[wi] = A[idxs, :].sum(axis=0)
    pooled = np.zeros((W, W), dtype=np.float32)
    for wj, idxs in enumerate(w2s):
        if idxs:
            pooled[:, wj] = rowpool[:, idxs].sum(axis=1)
    if normalize_rows:
        rs = pooled.sum(axis=1, keepdims=True)
        pooled = np.divide(pooled, np.maximum(rs, 1e-9))
    return pooled


def update_metrics_for_relation(
    acc1_sum, acck_sum, mass_sum, cnt_sum,
    A_LH_WW, heads, upos, deprel,
    target_rel: str, topk: int, exclude_punct: bool,
):
    """Accumulate metrics for one relation in-place."""
    L, H, W, _ = A_LH_WW.shape
    for i in range(W):  # dependent index
        if deprel[i] != target_rel:
            continue
        if exclude_punct and (upos[i] == "PUNCT"):
            continue
        h = heads[i]
        if h < 0 or h == i or h >= W:
            continue
        for l in range(L):
            for hh in range(H):
                row = A_LH_WW[l, hh, i]
                # top-1
                pred = int(np.argmax(row))
                acc1_sum[l, hh] += 1.0 if pred == h else 0.0
                # top-k
                if topk > 1:
                    topk_idx = np.argpartition(row, -topk)[-topk:]
                    acck_sum[l, hh] += 1.0 if h in topk_idx else 0.0
                else:
                    acck_sum[l, hh] += 1.0 if pred == h else 0.0
                # mass at gold head
                mass_sum[l, hh] += float(row[h])
                cnt_sum[l, hh] += 1.0


def main():
    ap = argparse.ArgumentParser(description="Relation-specific attention head probes over UD data")
    ap.add_argument("--attn-pkl", required=True, help="Pickle with examples from extract_attn.py")
    ap.add_argument("--model-id", required=True, help="HF tokenizer to rebuild subword mapping")
    ap.add_argument("--relations", default="amod,nsubj,obj,det,case,compound,advmod,obl",
                    help="Comma-separated UD relations to probe")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--exclude-punct", action="store_true")
    ap.add_argument("--outfile", required=True, help="Where to write JSON results")
    args = ap.parse_args()

    rels = [r.strip() for r in args.relations.split(",") if r.strip()]

    with open(args.attn_pkl, "rb") as f:
        data = pickle.load(f)

    tok = AutoTokenizer.from_pretrained(args.model_id)

    first = next(ex for ex in data if ex.get("attns") is not None)
    L, H = first["attns"].shape[:2]

    results = {
        "layers": L,
        "heads": H,
        "n_examples": 0,
        "relations": {}
    }
    for r in rels:
        results["relations"][r] = {
            "acc_top1_sum": np.zeros((L, H), dtype=np.float64),
            "acc_topk_sum": np.zeros((L, H), dtype=np.float64),
            "mass_sum":     np.zeros((L, H), dtype=np.float64),
            "cnt":          np.zeros((L, H), dtype=np.float64),
            "n_sentences":  0,
            "n_tokens":     0,
        }

    for ex in tqdm(data, desc="Probing relations"):
        A = ex.get("attns")
        words = ex.get("words")
        upos = ex.get("upos")
        heads = ex.get("head")
        deprel = ex.get("deprel")
        if A is None or words is None or upos is None or heads is None or deprel is None:
            continue

        w2s = build_word_to_subword(tok, words)
        W = len(w2s)
        A_LH_WW = np.zeros((L, H, W, W), dtype=np.float32)
        for l in range(L):
            for h in range(H):
                A_LH_WW[l, h] = pool_TT_to_WW(A[l, h], w2s, normalize_rows=True)

        for r in rels:
            acc1_sum = results["relations"][r]["acc_top1_sum"]
            acck_sum = results["relations"][r]["acc_topk_sum"]
            mass_sum = results["relations"][r]["mass_sum"]
            cnt_sum  = results["relations"][r]["cnt"]
            before = cnt_sum.sum()
            update_metrics_for_relation(acc1_sum, acck_sum, mass_sum, cnt_sum,
                                        A_LH_WW, heads, upos, deprel,
                                        target_rel=r, topk=args.topk,
                                        exclude_punct=args.exclude_punct)
            after = cnt_sum.sum()
            results["relations"][r]["n_sentences"] += 1
            results["relations"][r]["n_tokens"] += int(after - before)

        results["n_examples"] += 1

    for r in rels:
        R = results["relations"][r]
        cnt = np.maximum(R["cnt"], 1.0)
        R["acc_top1"] = (R["acc_top1_sum"] / cnt).tolist()
        R["acc_topk@%d" % args.topk] = (R["acc_topk_sum"] / cnt).tolist()
        R["avg_head_mass"] = (R["mass_sum"] / cnt).tolist()
        for k in ("acc_top1_sum","acc_topk_sum","mass_sum","cnt"):
            del R[k]

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote relation probe results → {args.outfile}")


if __name__ == "__main__":
    main()