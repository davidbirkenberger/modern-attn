# Modern BERT Attention Analysis (HF, PyTorch)

This project replicates the core analyses of Clark et al. (2019) without TensorFlow 1.x. It uses Hugging Face `transformers` and runs on Apple Silicon (M1/M2) via PyTorch (CPU or MPS).

## Installation

### Prerequisites

Install `uv` (recommended) or use traditional pip:

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

```bash
# Clone the repository
git clone <repository-url>
cd modern-attn

# Install dependencies with uv (recommended)
uv sync

# Or with traditional pip
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional: Install spaCy models for additional analysis
uv run python -m spacy download en_core_web_sm  # English
uv run python -m spacy download de_core_news_md  # German
```

## Quick Start

### 1. Run the Analysis Pipeline

Choose one of the available pipelines:

```bash
# Using uv (recommended)
uv run python pipeline_en.py      # English BERT analysis
uv run python pipeline_de.py      # German BERT analysis  
uv run python pipeline_rifel.py   # Custom model analysis

# Or with traditional activation
source .venv/bin/activate
python pipeline_en.py
```

## Citation

If you use this code, please also cite the original work:

```bibtex
@article{clark2019does,
  title={What does BERT look at? An analysis of BERT's attention},
  author={Clark, Kevin and Khandelwal, Urvashi and Levy, Omer and Manning, Christopher D},
  journal={Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP},
  pages={276--286},
  year={2019}
}
```
