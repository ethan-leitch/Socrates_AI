# Socrates AI — Project Status

A from-scratch, small decoder-only transformer (GPT-style) trained on public-domain philosophy texts from Project Gutenberg. Not a general chatbot — expect short, stylistically philosophical-flavored output, not always coherent, similar in spirit to a toy nanoGPT-scale model rather than a real LLM.

This file exists as a durable snapshot of the project's state, kept current enough to paste into a fresh Claude Code session (e.g. when building the personal-website integration) without needing the full conversation history that produced it.

## Corpus

- **634 books**, pulled from two Gutenberg bookshelves: `Philosophy` (78 books) and the larger `Category: Philosophy & Ethics` (556 books) — English text only.
- ~256,005,413 clean characters, ~64M estimated tokens (56.2M train / 7.46M val after an ~10% held-out-whole-books split).
- Validation is held out as **whole books**, not a slice of the token stream — tests generalization to unseen material, not just memorized continuations.
- Source data lives in `data/` (gitignored): `raw_books/`, `clean_books/`, `metadata.json`, `gutenberg_catalog_cache/` (the local SQLite Gutenberg catalog).

## Tokenizer

- Custom-trained **byte-level BPE** (HuggingFace `tokenizers` library, not `transformers`), trained on the cleaned corpus itself — not a pretrained/borrowed vocabulary.
- `vocab_size=16000`. Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>` (books are wrapped in `<bos>`/`<eos>` before being concatenated for training).
- Saved as a single file: `data/tokenizer.json` (vocab + merges + config together). Load via `Tokenizer.from_file(path)`.
- Byte-level means it can encode any input text; `<unk>` essentially never fires in practice.

## Model architecture

Built from scratch in PyTorch — no HuggingFace model classes. Lives in `socrates_ai/model_prep.py` (`MiniTransformer`, plus `Translator` for the tokenizer lifecycle).

- `vocab_size=16000, d_model=256, n_layers=4, n_heads=4, block_size=128, dropout=0.1`
- Standard pre-norm transformer blocks: causal self-attention via `F.scaled_dot_product_attention(..., is_causal=True)`, GELU feedforward (4x expansion), weight-tied embedding/output head.
- **7,288,320 parameters total.**
- Context is capped at 128 tokens — no positional embedding beyond that, generation always crops to the most recent 128 tokens.
- CPU-only so far (no CUDA/MPS on the dev machine) — ~2.7-4.8s per training step depending on vocab size, which is why training happens across multiple sessions (see below), not one sitting.

## Training (`socrates_ai/training.py`, `TrainingSocrates`)

Takes an already-built `MiniTransformer` (architecture only, constructed separately) and owns everything about training it: optimizer, LR schedule, gradient clipping, loss history, checkpointing, and generation.

- **AdamW**, weight_decay=1e-2, gradient clipping at norm 1.0.
- **LR schedule**: linear warmup over the first `warmup_steps=200`, then cosine decay to `min_lr=3e-5` by `decay_target_steps=4000`. Keyed off the **global**, persistent step count (`self.step`), not the size of any individual `.train()` call — warmup only ever fires once across the model's entire training lifetime, and decay progresses smoothly no matter how many separate sessions it takes to get there. (This was a real bug we found and fixed — an earlier per-call-scoped version re-triggered warmup on every resume, collapsing the LR back toward zero each time.)
- **Checkpointing**: `save_checkpoint()` writes model weights, optimizer state (so Adam's momentum survives a resume, not just the weights), loss history, and step count together into one file under `data/checkpoints/`. Auto-saves at the end of every `.train()` call. `load_checkpoint(path=None)` defaults to the most recent checkpoint, and checks the model's architecture (every parameter's shape) against what's recorded in the checkpoint *before* attempting to load — fails with one clear error on a mismatch (e.g. an accidentally-changed `d_model`) instead of either a cryptic PyTorch trace or silently loading something wrong.
- **Generation**: `generate()` (token-id level, autoregressive, temperature sampling via `torch.multinomial`, sliding-window context crop, optional `<eos>` early stop) and `talk()` (string in, string out — handles encode/decode).

## Current training progress

A real checkpoint exists: **`data/checkpoints/step_400.pt`** (400 steps trained so far, not yet a full run).

```
step   1 | train 9.74 | val 9.74   (init, ~ln(16000)=9.68, as expected)
step 200 | train 6.28 | val 6.35   (end of warmup)
step 400 | train 5.60 | val 5.64
```

Train/val tracking closely, no overfitting signal yet. This is early — nowhere near converged. Chinchilla-ratio math suggests something like ~2,000-4,000+ steps to start seeing meaningfully better output, and full convergence is further out than that. **No claims should be made yet about output quality** — treat any current checkpoint as a work-in-progress artifact, not a finished model.

## Package structure

```
socrates_ai/
  data_prep.py            TheFarmer (download+clean pipeline), EDAPainter (corpus EDA plots)
  helpers/
    data_helpers.py        low-level download/clean helpers, CleanedDataChecks
    resources.py            SQL query constants (PHILOSOPHY_QUERY, PHILOSOPHY_SHELVES)
    training_helpers.py     load_book_paths, tokenise_book, split_books_train_val
  model_prep.py            Translator (tokenizer lifecycle), MiniTransformer (architecture)
  training.py              TrainingSocrates (training orchestration)
```

Installed as an editable package via `pyproject.toml` + `uv` — `import socrates_ai...` works from anywhere in the repo (notebooks, scripts), no path hacks needed. Dependency versions are pinned carefully around real constraints: `torch==2.2.2` (last version with Intel Mac wheels), `numpy<2` / `pandas>=2.2,<3` (torch 2.2.2 predates NumPy 2.0 support).

## For the website integration specifically

- **Target interface**: `model = TrainedModel(load_path); output = model.generate(prompt, ...)` — this repo already has the equivalent (`TrainingSocrates.talk(tokeniser, prompt, ...)`), just needs wrapping for a standalone inference-only context (no training loop needed there).
- **What to bring over**: the `MiniTransformer` class (`model_prep.py`), a trained checkpoint (`.pt` file — `step_400.pt` exists but is early-stage; a more-trained one should replace it before this goes live), and `tokenizer.json`.
- **Not there yet**: a checkpoint actually trained to convergence. Don't wire up the website against `step_400.pt` expecting good output — it's a proof-that-the-pipeline-works artifact, not a finished model.
