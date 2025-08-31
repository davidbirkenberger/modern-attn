# run_pipeline_en.py
# Comments are in English.
import subprocess
import sys
import os

# --- Paths (edit to your local setup) ---
# Path to your UD English EWT .conllu file (dev set is fine to start)
CONLLU = "UD_English-EWT-master/en_ewt-ud-dev.conllu"   # <-- change if needed

# Output artifact paths
JSON   = "data/en_ewt_dev.json"
ATTN   = "out/en_ewt_attn.pkl"
REL    = "out/en_relations.json"

# Hugging Face model to analyze (English BERT)
MODEL  = "bert-base-uncased"  # or "bert-base-cased"

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("out", exist_ok=True)

# --- Step 1: UD (CoNLL-U) → JSON (words/upos/head/deprel/text) ---
subprocess.run([
    sys.executable, "ud_loader.py",
    "--input", CONLLU,
    "--output", JSON,
    "--limit", "500"  # adjust or remove to use full file
], check=True)

# --- Step 2: Extract attentions from BERT (writes [L,H,T,T]) ---
subprocess.run([
    sys.executable, "extract_attn.py",
    "--model-id", MODEL,
    "--input-json", JSON,
    "--output-pkl", ATTN
], check=True)

# --- Step 3: Relation-specific probes (find specialized heads) ---
subprocess.run([
    sys.executable, "relation_probes.py",
    "--attn-pkl", ATTN,
    "--model-id", MODEL,
    "--relations", "amod,nsubj,obj,det,case",
    "--topk", "3",
    "--exclude-punct",
    "--outfile", REL
], check=True)

print("All done. Results written to:", REL)