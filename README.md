# Modern BERT Attention Analysis (HF, PyTorch)

This project replicates the core analyses of Clark et al. (2019) without TensorFlow 1.x. It uses Hugging Face `transformers` and runs on Apple Silicon (M1/M2) via PyTorch (CPU or MPS).

## 1) Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
# Add a spaCy model (choose one):
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_md