# Comments are in English. Computes JS-distance between per-head signatures across corpus.
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from utils_io import read_pickle

SPECIAL = {"[CLS]", "[SEP]", "[PAD]"}


def head_signature(A, tokens):
    T = A.shape[0]
    keep = [i for i, t in enumerate(tokens) if t not in SPECIAL]
    if not keep:
        p = A.flatten()
    else:
        A2 = A[np.ix_(keep, keep)]
        p = A2.flatten()
    s = float(p.sum())
    return p / s if s > 0 else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-pkl", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    data = read_pickle(args.attn_pkl)
    L = H = None
    sigs, counts = None, None

    for ex in tqdm(data, desc="Accumulating head signatures"):
        attn = ex["attns"]
        tokens = ex.get("tokens") or ex.get("words")
        if tokens is None:
            continue
        if L is None:
            L, H = attn.shape[:2]
            sigs = [[None for _ in range(H)] for _ in range(L)]
            counts = np.zeros((L, H), dtype=np.int32)
        for l in range(L):
            for h in range(H):
                p = head_signature(attn[l, h], tokens)
                if sigs[l][h] is None:
                    sigs[l][h] = np.zeros_like(p, dtype=np.float64)
                sigs[l][h] += p
                counts[l, h] += 1

    for l in range(L):
        for h in range(H):
            if counts[l, h] > 0:
                sigs[l][h] /= counts[l, h]
            else:
                sigs[l][h] = np.zeros_like(sigs[l][h])

    flat = [sigs[l][h] for l in range(L) for h in range(H)]
    LH = len(flat)
    Dmat = np.zeros((LH, LH), dtype=np.float32)
    for i in range(LH):
        for j in range(i + 1, LH):
            d = jensenshannon(flat[i], flat[j])
            Dmat[i, j] = Dmat[j, i] = float(d)

    np.save(args.outfile, Dmat)
    print(f"Saved JS distance matrix: {args.outfile} (shape={Dmat.shape})")


if __name__ == "__main__":
    main()