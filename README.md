# WikiGPT-Mini

A small GPT-style language model built from scratch in PyTorch, trained on Wikipedia text with a custom SentencePiece tokenizer.

## Description

WikiGPT-Mini is an educational, decoder-only Transformer implementation designed to be readable and easy to experiment with. It includes the full pipeline in a single script: tokenization, data loading, model definition, training with checkpointing, and interactive text generation.

The attention mechanism is implemented manually (per-head query/key/value projections with causal masking), and the script also contains step-by-step cells that help you inspect and understand how the attention matrix and value updates work.

## Features

- Decoder-only Transformer (GPT-style) with causal self-attention
- Multi-head attention implemented from scratch
- Pre-LayerNorm blocks with residual connections and GELU feed-forward layers
- SentencePiece tokenizer support
- Cached encoded dataset (`encoded_data.pt`) for faster restarts
- AdamW optimizer with selective weight decay, cosine learning-rate schedule and gradient clipping
- Training in `bfloat16` with TF32 matmul enabled
- Checkpoint saving/loading (best validation loss)
- Optional Weights & Biases logging
- Interactive inference mode

## Default Configuration

| Parameter        | Value  |
|------------------|--------|
| Context length   | 512    |
| Embedding size   | 384    |
| Layers           | 7      |
| Attention heads  | 7      |
| Batch size       | 32     |
| Learning rate    | 3e-4   |
| Dropout          | 0.05   |
| Weight decay     | 0.01   |
| Gradient clip    | 1.0    |
| Training iters   | 100,000 |

All values can be changed at the top of the script.

## Requirements

- Python 3.9+
- PyTorch (a CUDA GPU is strongly recommended)
- `sentencepiece`, `tqdm`, `wandb`, `ipdb`, `requests`

```bash
pip install torch sentencepiece tqdm wandb ipdb requests
```

## Project Files

You need to provide the following files in the working directory:

| File                  | Description                                         |
|-----------------------|-----------------------------------------------------|
| `wiki.txt`            | Plain-text training corpus (e.g. Wikipedia dump)    |
| `wiki_tokenizer.model`| Trained SentencePiece model                         |

Generated automatically:

| File                  | Description                                         |
|-----------------------|-----------------------------------------------------|
| `encoded_data.pt`     | Cached tokenized dataset                            |
| `models/latest.pt`    | Latest best checkpoint (model + optimizer state)    |

> Make sure the `models/` directory exists before training.

## Usage

### Training

1. Set `inference = False` in the script.
2. (Optional) Set `wandb_log = False` if you don't want to use Weights & Biases.
3. Run the script:

```bash
python "Large Language Model.py"
```

Training resumes automatically from `models/latest.pt` when `load_pretrained = True` and the checkpoint exists.

### Inference

1. Set `inference = True` and make sure a checkpoint exists in `models/`.
2. Run the script and type a prompt when asked:

```
Enter text (q to quit): Once upon a time
```

Type `q` to quit.

## Architecture Overview

```
Token IDs
   │
Token Embedding + Positional Embedding
   │
[ Transformer Block ] x N
   │   ├─ LayerNorm → Multi-Head Causal Self-Attention → Residual
   │   └─ LayerNorm → Feed-Forward (GELU) → Residual
   │
LayerNorm
   │
Linear → Vocabulary Logits
```

## Notes

- The script was exported from a Google Colab notebook, so some cells (such as `!pip install`) only work in a notebook environment. Remove or replace them when running as a plain `.py` file.
- Text generation uses plain multinomial sampling (no temperature or top-k yet).

## License

Choose a license for your project (for example, MIT) and add a `LICENSE` file.
