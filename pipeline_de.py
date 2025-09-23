import subprocess
import sys

# Paths
CONLLU = "UD_German-GSD-master/de_gsd-ud-dev.conllu"
JSON   = "data/de_gsd_dev.json"
ATTN   = "out/de_gsd_attn.pkl"
REL    = "out/de_relations.json"
MODEL  = "dbmdz/bert-base-german-cased"

# Step 1: UD → JSON
subprocess.run([
    sys.executable, "ud_loader.py",
    "--input", CONLLU,
    "--output", JSON,
    "--limit", "500"
], check=True)

# Step 2: Extract attentions
subprocess.run([
    sys.executable, "extract_attn.py",
    "--model-id", MODEL,
    "--input-json", JSON,
    "--output-pkl", ATTN
], check=True)

# Step 3: Relation probes
subprocess.run([
    sys.executable, "relation_probes.py",
    "--attn-pkl", ATTN,
    "--model-id", MODEL,
    "--relations", "amod,nsubj,obj,det,case",
    "--topk", "3",
    "--exclude-punct",
    "--outfile", REL
], check=True)

print("All done. Results in:", REL)