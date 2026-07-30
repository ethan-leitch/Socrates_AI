# Socrates AI

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Status: In Development](https://img.shields.io/badge/status-in%20development-yellow.svg)

An end to end implementation of a GPT-style language model, trained from scratch on 634 gutenberg philosophy books.

## Table of Contents
- [Status](#status)
- [Overview](#overview)
- [Code Layout](#code-layout)
- [Data & Tokeniser](#data--tokeniser)
- [Architecture & Training](#architecture--training)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Example Output](#example-output)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

## Status
Currently in development, general architecture is laid out, and training + parameter tuning is in progress

## Overview
From data collection, analysis, cleaning, pre processing, model training, storage and inference, this project aims to explore the whole process from start to finish of building a language model without relying on an already pretrained model. I also had a focus on using software engineering practices, using OOP to practice building scaleable systems that can be used in production enviornments.

Instead of only focusing on the transformer, the goal of this project is to also have an end to end structure that can be easily downloaded and used or placed in a framework and not run into any issues.

This project is inspired from my 3rd year ML module where I built a character level language model using a stacked GRU architecture. However the courework was mainly focused on understanding the underlying ML concepts, and was limited to one notebook so there wasn't much opportunity to try out data engineering practices, package design or production workflows. Becuase of this, I've aimed to design this project in a more production style setting


## Code Layout
Pre-processing:
[`TheFarmer`](socrates_ai/data_prep.py#L21) -> Everything to do with the raw .txt book files. Ingests the training data, sets correct directories for storing files, wraps the GutenbergCache class (more info below) to actually retrieve the books, uses SQL queries to filter from the Gutenberg STAR schema and cleans and runs tests on the books. Also creates a metadata file which contains useful info about the books we needed.

-- [`CleanedDataChecks`](socrates_ai/helpers/data_helpers.py#L118) -> A simple class to group all of the data checks together, used in `TheFarmer`.

[`EDAPainter`](socrates_ai/data_prep.py#L171) -> All data analysis visuals can be easily produced using this class, automatically uses the exact file paths created in the farmer, but the user has the option to manually set them too. Creates a nice interpretable dataframe from the metadata json. `EDAPainter` also has style, setting each plot to a custom aesthetic colour scheme.

Model Preparation:
[`Translator`](socrates_ai/model_prep.py#L14) -> Uses the classic Byte Pair Encoding algorithm to tokenise the books. Vocab size is set to 16000 but can be set by the user. Creates an easy interface for the user to tokenise all the training data. Saves the tokenised books in a structured directory, as well as functionality to load the trained tokeniser and test it for a quick sanity check

[`MiniTransformer`](socrates_ai/model_prep.py#L147) -> Implements the GPT-style transformer architecture using PyTorch. Uses [`CausalSelfAttention`](socrates_ai/model_prep.py#L69), [`FeedForward`](socrates_ai/model_prep.py#L110) and [`TransformerBlock`](socrates_ai/model_prep.py#L128) to construct the final structure. More details on the framework and specific parameters will be laid out in "(more_info tbc, I will come back to this)"

Training:
[`TrainingSocrates`](socrates_ai/training.py#L11) -> This is where our Socrates AI learns, aka the model training functionality. Includes batching, loss estimates, learning rate warmup + decay, gradient clipping and auto checkpoint saving of the model during training. The user can easily then reload and partly trained model simply using the `load_checkpoint()` method. Alongside this it provides us with a simple plot of loss curves and basic inference using the `.talk()` method.

## Data & Tokeniser
All data used to train the model is from the Gutenberg project, the oldest digital library. (https://www.gutenberg.org/about/). I've filtered for all books in the philosophy category, which totals to 634 books, and roughly 63.5 million tokens. An important note is that all of the data is not seen in the github repo due to the size (it's in .gitignore), but if you were to run the farmer it would download in the correct places and create the file structure.

I decided to go with Hugging Face's tokenizers library to actually implement the tokenising logic, it's pretty simple to use does the job, and importantly lets us train the tokeniser on our entire corpus. This is useful as we're likely to have a more specific set of tokens to do with philosophy, rather than use a pre-trained one where it has more standard tokens.


## Architecture & Training

Model: `vocab_size=16000, d_model=256, n_layers=4, n_heads=4, block_size=128, dropout=0.1` — **7,288,320 parameters** total.

Optimiser is AdamW with gradient clipping (norm 1.0), and the learning rate follows a linear warmup (200 steps) into a cosine decay down to a minimum LR. Warmup only ever happens once across the model's whole training history — not on every resume — since resuming restores the optimiser's own momentum state too, so there's nothing left for a fresh warmup to protect against.

`TrainingSocrates` checkpoints automatically during training (not just at the end of a run), saving model weights, optimiser state, loss history and step count together into one file. This means training can be safely stopped and resumed across multiple sessions — `load_checkpoint()` picks up exactly where the last run left off, including the LR schedule position, and validates the model's architecture against the checkpoint before loading so a shape mismatch fails loudly instead of silently.

## Project Structure

```
Socrates_AI/
├── socrates_ai/                  # installable package (uv/pyproject.toml)
│   ├── data_prep.py               # TheFarmer (download+clean pipeline), EDAPainter (corpus EDA plots)
│   ├── model_prep.py              # Translator (tokeniser lifecycle), MiniTransformer (model architecture)
│   ├── training.py                # TrainingSocrates (training loop, checkpointing, generation)
│   └── helpers/
│       ├── data_helpers.py        # low-level download/clean helpers, CleanedDataChecks
│       ├── resources.py           # SQL query constants (PHILOSOPHY_QUERY, PHILOSOPHY_SHELVES)
│       └── training_helpers.py    # load_book_paths, tokenise_book, split_books_train_val
├── notebooks/
│   ├── pre_processing.ipynb       # data download, cleaning, EDA
│   ├── model_creation.ipynb       # tokeniser + model build + training
│   └── project_runthrough.ipynb   # end-to-end walkthrough (in progress)
├── data/                          # gitignored — created locally by TheFarmer
├── pyproject.toml
└── README.md
```


## Setup & Usage

Please see runthrough.ipynb (to be finished soon)

## Example Output

Prompt: `"The greatest happiness"` (temperature 0.85)

```
The greatest happiness of the decapore man's overgold and the love of the
soul and the rest, and the make made to our power.

This is divine, the head of the world, the path of the soul:

A Soul, andols, the individual he has power. But when we profound the
dark cloud of the soul, we are, and our man must keep a world with the
decency of the soul.
```

As you can see, this is very much a work in progress, grammar and philosophical vocabulary come through, but coherence at the sentence-to-sentence level doesn't yet. See the [Roadmap](#roadmap) below for where this stands.

## Roadmap

I am currently trying to get this model to produce somewhat readable outputs. I've done some initial training and while it did output sentences, they had little meaning. Initial experiments (~6 hours of training) had a stable loss decrease, but the loss curve flattened out quickly, so I'm experimenting with setting a higher step at which the learning rate decays to, as well as changing dropout from 0.1 to 0.05.

Ultimately I'm not expecting this to produce high quality readable sentences, but I am hoping for a kind of philosophical style to the sentences and some interesting phrases. Even though we have 634 books, for a transformer based language model, this is still a very small training dataset, and additionally this model only has about 7.3M parameters. To have a point of comparison, GPT 2 had 124M (17x bigger), and GPT 3 had 175B (24,000x bigger!). 


## Acknowledgments

Will be filled in soon