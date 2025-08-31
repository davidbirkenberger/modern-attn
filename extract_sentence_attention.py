#!/usr/bin/env python3
"""
Extract attention for a specific sentence from specified heads of the rifel model at the word level.
Uses the pool_TT_to_WW method from relation_probes.py to convert token-level attention to word-level.
"""

import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple


from relation_probes import build_word_to_subword, pool_TT_to_WW


def extract_sentence_attention(
    model_id: str,
    sentence: str,
    words: List[str],
    target_layer_heads: List[Tuple[int, int]],
    max_length: int = 128
) -> dict:
    """
    Extract attention for a specific sentence from specified heads at word level.
    
    Args:
        model_id: Path to the rifel model
        sentence: The sentence to analyze
        words: List of words in the sentence
        target_heads: List of head indices to extract attention from
        max_length: Maximum sequence length
    
    Returns:
        Dictionary containing attention matrices and metadata
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_attentions=True).to(device).eval()
    
    # Encode the sentence
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    
    # Get attention
    with torch.no_grad():
        res = model(**enc)
    
    # Extract attention matrices
    attn = torch.stack([a[0] for a in res.attentions]).to("cpu").to(torch.float32).numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].to("cpu").tolist())
    
    # Get word-to-subword mapping
    w2s = build_word_to_subword(tokenizer, words)
    
    # Extract attention for specified layer-head combinations
    L, H, T, _ = attn.shape
    W = len(w2s)
    
    print(f"Model has {L} layers and {H} heads")
    print(f"Sentence has {W} words and {T} tokens")
    print(f"Target layer-head combinations: {target_layer_heads}")
    
    # Validate layer-head combinations
    valid_combinations = []
    for l, h in target_layer_heads:
        if 0 <= l < L and 0 <= h < H:
            valid_combinations.append((l, h))
        else:
            print(f"Warning: Invalid layer-head combination L{l}-H{h} will be ignored")
    
    # Extract attention for each specified layer and head
    results = {
        "sentence": sentence,
        "words": words,
        "tokens": tokens,
        "word_to_subword": w2s,
        "layers": L,
        "total_heads": H,
        "target_layer_heads": valid_combinations,
        "attention_matrices": {}
    }
    
    for l, h in valid_combinations:
            # Get token-level attention for this layer and head
            A_TT = attn[l, h]
            
            # Convert to word-level attention
            A_WW = pool_TT_to_WW(A_TT, w2s, normalize_rows=True)
            
            key = f"L{l}_H{h}"
            results["attention_matrices"][key] = {
                "layer": l,
                "head": h,
                "token_attention": A_TT.tolist(),
                "word_attention": A_WW.tolist(),
                "word_attention_shape": A_WW.shape
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention for a specific sentence from specified heads at word level"
    )
    parser.add_argument(
        "--model-id", 
        default="./rifel_multilabel",
        help="Path to the rifel model (default: ./rifel_multilabel)"
    )
    parser.add_argument(
        "--sentence", 
        default="Wir bieten exklusives Catering",
        help="Sentence to analyze (default: 'Wir liefern das Catering.')"
    )
    parser.add_argument(
        "--layer-heads", 
        nargs="+", 
        default=["0-0", "1-1", "2-2", "3-3", "4-4", "5-5"],
        help="Layer-head combinations to extract attention from in format 'L-H' (default: 0-0 1-1 2-2 3-3 4-4 5-5)"
    )
    parser.add_argument(
        "--output", 
        default="/Users/davidbirkenberger/Projects/mechinterp_article/public/sentence_attention.json",
        help="Output JSON file (default: sentence_attention.json)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    
    args = parser.parse_args()
    
    # Parse layer-head combinations
    layer_heads = []
    for lh_str in args.layer_heads:
        try:
            l, h = map(int, lh_str.split('-'))
            layer_heads.append((l, h))
        except ValueError:
            print(f"Warning: Invalid layer-head format '{lh_str}', expected 'L-H'")
            continue
    
    if not layer_heads:
        print("Error: No valid layer-head combinations provided")
        return 1
    
    # Split sentence into words
    words = args.sentence.split()
    
    print(f"Analyzing sentence: '{args.sentence}'")
    print(f"Words: {words}")
    print(f"Target layer-head combinations: {layer_heads}")
    
    try:
        results = extract_sentence_attention(
            model_id=args.model_id,
            sentence=args.sentence,
            words=words,
            target_layer_heads=layer_heads,
            max_length=args.max_length
        )
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {args.output}")
        
        # Print summary
        print(f"\nExtracted attention for {len(results['attention_matrices'])} layer-head combinations:")
        for key, data in results['attention_matrices'].items():
            print(f"  {key}: Layer {data['layer']}, Head {data['head']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
