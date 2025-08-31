# Comments are in English. Computes governor hit-rate per head using spaCy parses.
import argparse
import json
import numpy as np
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer
from utils_io import read_pickle


def build_word_to_subword_spacy(tokenizer, doc):
    enc = tokenizer(doc.text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    word_to_sub = []
    for tok in doc:
        ws, we = tok.idx, tok.idx + len(tok.text)
        idxs = []
        for i, (a, b) in enumerate(offsets):
            if a == b:
                continue
            if not (b <= ws or a >= we):
                idxs.append(i)
        word_to_sub.append(idxs)
    return word_to_sub


def pool_matrix_wordlevel(A, word_to_sub):
    W = len(word_to_sub)
    row_blocks = [np.mean(A[idxs, :], axis=0) if idxs else np.zeros(A.shape[1]) for idxs in word_to_sub]
    pooled = np.zeros((W, W), dtype=A.dtype)
    for wi in range(W):
        pooled[wi] = [np.mean(row_blocks[wi][cols]) if cols else 0.0 for cols in word_to_sub]
    return pooled


def governor_hit_rate_for_example(doc, A_LH_TT, word_to_sub):
    L, H, T, _ = A_LH_TT.shape
    W = len(word_to_sub)
    scores = np.zeros((L, H), dtype=np.float32)
    counts = np.zeros((L, H), dtype=np.int32)
    for l in range(L):
        for h in range(H):
            A = A_LH_TT[l, h]
            Wmat = pool_matrix_wordlevel(A, word_to_sub)
            for i, tok in enumerate(doc):
                if i >= W or tok.is_punct:
                    continue
                head_idx = tok.head.i
                if head_idx == i:
                    continue
                pred = int(np.argmax(Wmat[i]))
                scores[l, h] += int(pred == head_idx)
                counts[l, h] += 1
    acc = np.divide(scores, np.maximum(counts, 1))
    return acc, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-pkl", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--spacy-model", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    nlp = spacy.load(args.spacy_model)
    data = read_pickle(args.attn_pkl)

    head_scores = None
    n_examples = 0

    for ex in tqdm(data, desc="Probing heads"):
        text = ex.get("text", None)
        attn = ex.get("attns", None)
        if text is None or attn is None:
            continue
        doc = nlp(text)
        w2s = build_word_to_subword_spacy(tok, doc)
        acc, cnt = governor_hit_rate_for_example(doc, attn, w2s)
        if head_scores is None:
            head_scores = {"sum": acc, "cnt": cnt}
        else:
            head_scores["sum"] += acc
            head_scores["cnt"] += cnt
        n_examples += 1

    mean_acc = np.divide(head_scores["sum"], np.maximum(head_scores["cnt"], 1))
    out = {"n_examples": n_examples, "mean_accuracy": mean_acc.tolist()}
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote probe results for {n_examples} examples → {args.outfile}")


if __name__ == "__main__":
    main()