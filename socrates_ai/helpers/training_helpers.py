import glob
import os
import random

import torch


def load_book_paths(clean_dir):
    # sorted() so the file order is deterministic across runs -- without it,
    # glob's order isn't guaranteed, which would make the train/val split
    # below silently different every time we re-run this notebook
    return sorted(glob.glob(os.path.join(clean_dir, "*.txt")))


def tokenise_book(path, tokeniser):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = tokeniser.encode(text).ids

    bos_id = tokeniser.token_to_id("<bos>")
    eos_id = tokeniser.token_to_id("<eos>")
    # wrap each book in <bos>/<eos> before concatenating books together later --
    # without this the model would see one book's ending flow straight into the
    # next book's opening with no signal that anything changed
    return [bos_id] + ids + [eos_id]


def split_books_train_val(book_paths, tokeniser, val_fraction=0.1, seed=231103):
    # fixed seed -> same split every time we re-run the notebook, so results
    # are comparable across experiments instead of validating against a
    # different random slice of books each run
    rng = random.Random(seed)
    shuffled = book_paths.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    # held-out WHOLE books, not a slice cut out of one long token stream --
    # this tests whether the model generalises to material it has never seen
    # any part of, rather than just memorising continuations of familiar text
    #
    # NOTE: the split itself (above) is effectively instant -- the real cost
    # of this function is what follows: tokenising the full text of every
    # book. Measured at ~5 minutes across 634 books, scales with corpus size.
    train_ids = [tid for path in train_paths for tid in tokenise_book(path, tokeniser)]
    val_ids = [tid for path in val_paths for tid in tokenise_book(path, tokeniser)]

    # long tensor because these are token ids (indices into the embedding
    # table), not continuous values
    return (
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(val_ids, dtype=torch.long),
    )
