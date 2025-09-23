#!/usr/bin/env python3
"""
Head ablation via forward hooks + pseudo-perplexity (MLM leave-one-out).

Usage example:

python ablate_heads_ppl.py \
  --model-id dbmdz/bert-base-german-cased \
  --conllu UD_German-GSD-master/de_gsd-ud-dev.conllu \
  --heads-json heads_de.json \
  --relations amod,nsubj,obj \
  --filter-by-relation \
  --max-length 256 \
  --device cpu \
  --n-random 5 \
  --outfile out/ablation_de.json

The heads JSON is a mapping relation -> list[[layer_idx, head_idx]], e.g.:
{"nsubj": [[8,3],[9,7]], "obj": [[8,6]], "amod": [[6,9]]}
"""

from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import os

# Local helpers if available
try:
    from ud_loader import load_conllu
except Exception:
    load_conllu = None  # type: ignore

try:
    from utils_io import read_jsonl_or_json
except Exception:
    read_jsonl_or_json = None  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_texts(
    conllu: str | None,
    json_path: str | None,
    relation_filter: str | None,
    limit_sentences: int | None = None,
) -> List[Dict[str, object]]:
    """Load sentences as dicts with at least 'text'. If available, keep 'deprel'.
    Returns a list of dicts with fields: text: str, deprel?: list[str]
    If relation_filter is provided, keep only sentences containing that deprel.
    """
    items: List[Dict[str, object]] = []
    if conllu:
        if load_conllu is None:
            raise RuntimeError("ud_loader.load_conllu not found; run within project root")
        sents = load_conllu(conllu)
        for s in sents:
            items.append({
                "text": s.get("text", ""),
                "deprel": s.get("deprel", []),
            })
    elif json_path:
        if read_jsonl_or_json is None:
            # Fallback: minimal JSON reader
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        else:
            data = read_jsonl_or_json(json_path)
        for x in data:
            if isinstance(x, str):
                items.append({"text": x})
            elif isinstance(x, dict):
                txt = x.get("text")
                if not txt and "words" in x:
                    # reconstruct if necessary
                    txt = " ".join(x.get("words", []))
                items.append({"text": txt or "", "deprel": x.get("deprel")})
            else:
                continue
    else:
        raise ValueError("Provide either --conllu or --json")

    if relation_filter:
        kept: List[Dict[str, object]] = []
        for s in items:
            deps = s.get("deprel") if isinstance(s, dict) else None
            if isinstance(deps, list) and any(d == relation_filter for d in deps):
                kept.append(s)
        if kept:
            items = kept
        else:
            print(f"[warn] --filter-by-relation active but no matching '{relation_filter}' found; using full set")
    if limit_sentences is not None:
        items = items[: max(0, int(limit_sentences))]
    return items


def pseudo_ppl(model, tok, texts: Sequence[str], max_len: int = 256, device: str = "cpu") -> float:
    """Leave-one-out negative log-likelihood averaged over tokens.
    Returns mean NLL per token (pseudo-perplexity proxy).
    """
    model.eval()
    total_nll: float = 0.0
    total_tokens: int = 0
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no [MASK] token; use an MLM model")
    special_ids = set(tok.all_special_ids or [])

    with torch.no_grad():
        for s in texts:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = enc["input_ids"][0].to(device)
            attn_mask = enc.get("attention_mask")
            attn_mask = attn_mask[0].to(device) if attn_mask is not None else None
            L = input_ids.size(0)
            for i in range(L):
                tid = int(input_ids[i].item())
                if tid in special_ids:
                    continue
                masked = input_ids.clone()
                masked[i] = mask_id
                if attn_mask is not None:
                    out = model(input_ids=masked.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0))
                else:
                    out = model(input_ids=masked.unsqueeze(0))
                logit = out.logits[0, i]
                nll = torch.nn.functional.cross_entropy(
                    logit.unsqueeze(0), torch.tensor([tid], device=logit.device)
                )
                total_nll += float(nll)
                total_tokens += 1
    return total_nll / max(1, total_tokens)


# --- Head ablation hooks ---

def make_head_zero_hook(head_indices: set[int]):
    def hook(module, inputs, output):
        # output may be Tensor or tuple
        if isinstance(output, tuple):
            context = output[0]
        else:
            context = output
        # context [B, T, H]
        B, T, H = context.shape
        nh = getattr(module, "num_attention_heads", None)
        if nh is None:
            # Fallback: infer from hidden size and head dim
            head_dim = getattr(module, "attention_head_size", None)
            if head_dim is None:
                raise RuntimeError("Cannot infer number of heads from module: missing num_attention_heads/attention_head_size")
            nh = H // int(head_dim)
        hd = H // int(nh)
        ctx = context.view(B, T, int(nh), hd)
        with torch.no_grad():
            for h in head_indices:
                if 0 <= h < int(nh):
                    ctx[:, :, h, :] = 0
        context_z = ctx.view(B, T, H)
        if isinstance(output, tuple):
            return (context_z,) + output[1:]
        else:
            return context_z
    return hook


def get_self_attention_module(model, layer_idx: int):
    # Try BERT
    try:
        return model.bert.encoder.layer[layer_idx].attention.self
    except Exception:
        pass
    # Try RoBERTa/XLM-R
    try:
        return model.roberta.encoder.layer[layer_idx].attention.self
    except Exception:
        pass
    # Try base_model
    try:
        return model.base_model.encoder.layer[layer_idx].attention.self
    except Exception:
        pass
    raise AttributeError("Could not locate encoder.layer[{}].attention.self".format(layer_idx))


def register_ablation_hooks(model, layer_to_heads: Dict[int, List[int]]):
    handles = []
    for l, heads in layer_to_heads.items():
        sa = get_self_attention_module(model, l)
        h = sa.register_forward_hook(make_head_zero_hook(set(heads)))
        handles.append(h)
    return handles


def parse_heads_json(path_or_str: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parse a mapping relation -> list[[layer, head]]. Accept file path or JSON string."""
    if Path(path_or_str).exists():
        data = json.loads(Path(path_or_str).read_text(encoding="utf-8"))
    else:
        data = json.loads(path_or_str)
    out: Dict[str, List[Tuple[int, int]]] = {}
    for k, v in data.items():
        pairs: List[Tuple[int, int]] = []
        for p in v:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pairs.append((int(p[0]), int(p[1])))
        if pairs:
            out[k] = pairs
    return out


def group_by_layer(pairs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    d: Dict[int, List[int]] = defaultdict(list)
    for l, h in pairs:
        d[l].append(h)
    return dict(d)


def sample_layer_matched_random(layer_to_heads: Dict[int, List[int]], num_heads: int) -> Dict[int, List[int]]:
    """Sample per-layer the same count of heads, without replacement, excluding specified heads.
    num_heads is the model's num_attention_heads.
    """
    out: Dict[int, List[int]] = {}
    for l, hs in layer_to_heads.items():
        k = len(hs)
        pool = [h for h in range(num_heads) if h not in set(hs)]
        if k > len(pool):
            # fallback: allow sampling with replacement
            out[l] = random.choices(pool, k=k)
        else:
            out[l] = random.sample(pool, k=k)
    return out


def _load_with_token(load_fn, model_id: str, token: str | None, **kwargs):
    """Helper to load HF resources handling older/newer token kwarg names."""
    if token:
        try:
            return load_fn(model_id, token=token, **kwargs)
        except TypeError:
            return load_fn(model_id, use_auth_token=token, **kwargs)
    else:
        return load_fn(model_id, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="Ablate attention heads and measure pseudo-perplexity deltas")
    ap.add_argument("--model-id", required=True, help="HF model id for MLM (e.g., bert-base-german-cased)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--conllu", help="Path to UD *.conllu for sentence texts")
    src.add_argument("--json", help="Path to JSON list/JSONL with items having 'text' (optional 'deprel')")
    ap.add_argument("--heads-json", required=True, help="Path or JSON string mapping relation->[[layer,head],...]")
    ap.add_argument("--relations", default=None, help="Comma-separated subset of relations to evaluate; default: all in heads JSON")
    ap.add_argument("--filter-by-relation", action="store_true", help="For each relation, only evaluate sentences where it occurs")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-random", type=int, default=5, help="Number of layer-matched random control runs")
    ap.add_argument("--outfile", default=None, help="Where to write JSON report")
    ap.add_argument("--limit-sentences", type=int, default=None, help="Optional limit of sentences for quick runs")
    ap.add_argument("--hf-token", default=None, help="Hugging Face token for private models (falls back to $HF_TOKEN/$HUGGINGFACE_TOKEN)")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load model/tokenizer
    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    tok = _load_with_token(AutoTokenizer.from_pretrained, args.model_id, token)
    model = _load_with_token(AutoModelForMaskedLM.from_pretrained, args.model_id, token)
    device = torch.device(args.device)
    model.to(device)

    # Parse heads mapping
    rel2pairs = parse_heads_json(args.heads_json)
    if args.relations:
        wanted = {r.strip() for r in args.relations.split(",") if r.strip()}
        rel2pairs = {r: ps for r, ps in rel2pairs.items() if r in wanted}
    if not rel2pairs:
        raise SystemExit("No relations to evaluate after filtering")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    results = {
        "model": args.model_id,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_length": args.max_length,
        "device": str(device),
        "seed": args.seed,
        "n_random": args.n_random,
        "per_relation": {}
    }

    # Evaluate per relation
    for rel, pairs in rel2pairs.items():
        print(f"[info] Relation '{rel}': {len(pairs)} heads → {pairs}")
        layer_to_heads = group_by_layer(pairs)

        # Load/Filter texts for this relation
        texts_raw = load_texts(
            args.conllu,
            args.json,
            rel if args.filter_by_relation else None,
            limit_sentences=args.limit_sentences,
        )
        texts = [str(x.get("text", "")) for x in texts_raw]
        texts = [t for t in texts if t.strip()]
        if not texts:
            print(f"[warn] No texts available for relation '{rel}', skipping")
            continue

        print(f"[info] Computing baseline pseudo-ppl on {len(texts)} sentences …")
        ppl_base = pseudo_ppl(model, tok, texts, max_len=args.max_length, device=str(device))

        # Target ablation
        print(f"[info] Ablating target heads and recomputing pseudo-ppl …")
        handles = register_ablation_hooks(model, layer_to_heads)
        try:
            ppl_target = pseudo_ppl(model, tok, texts, max_len=args.max_length, device=str(device))
        finally:
            for h in handles:
                h.remove()
        delta = ppl_target - ppl_base

        # Random controls
        rand_deltas: List[float] = []
        print(f"[info] Running {args.n_random} random layer-matched controls …")
        for r_i in range(args.n_random):
            rand_map = sample_layer_matched_random(layer_to_heads, num_heads)
            handles = register_ablation_hooks(model, rand_map)
            try:
                ppl_rand = pseudo_ppl(model, tok, texts, max_len=args.max_length, device=str(device))
            finally:
                for h in handles:
                    h.remove()
            rand_deltas.append(ppl_rand - ppl_base)

        rand_mean = float(sum(rand_deltas) / max(1, len(rand_deltas)))
        rand_std = float(math.sqrt(sum((x - rand_mean) ** 2 for x in rand_deltas) / max(1, len(rand_deltas))))

        results["per_relation"][rel] = {
            "n_sentences": len(texts),
            "ppl_base": ppl_base,
            "ppl_target": ppl_target,
            "delta_target": delta,
            "delta_random_mean": rand_mean,
            "delta_random_std": rand_std,
            "heads": pairs,
        }

        print({
            "relation": rel,
            "ppl_base": ppl_base,
            "ppl_target": ppl_target,
            "delta": delta,
            "delta_random_mean": rand_mean,
            "delta_random_std": rand_std,
        })

    if args.outfile:
        Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[done] Wrote report → {args.outfile}")


if __name__ == "__main__":
    main()
