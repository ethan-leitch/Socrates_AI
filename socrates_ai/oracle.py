import glob
import os

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class Translator:
    def __init__(self, cleaned_books_path, Vocab_size = 16000):
        self.books_path = cleaned_books_path
        self.vocab_size = Vocab_size
        self.tokeniser_path = None

    def train_tokeniser(self):

        tokeniser = Tokenizer(BPE(unk_token="<unk>"))
        tokeniser.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokeniser.decoder = decoders.ByteLevel()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        corpus_files = sorted(glob.glob(os.path.join(self.books_path, "*.txt")))
        print(f"Training on {len(corpus_files)} files")
        tokeniser.train(files=corpus_files, trainer=trainer)
        tokeniser_path = os.path.join("..", "data", "tokenizer.json")
        tokeniser.save(tokeniser_path)

        print(f"Trained vocab size: {tokeniser.get_vocab_size()}")
        print(f"Saved to {tokeniser_path}")
        self.tokeniser_path = tokeniser_path

    def load_tokeniser(self, path=None):
        self.tokeniser_path = path if path is not None else self.tokeniser_path

        if self.tokeniser_path is None:
            raise ValueError("Tokeniser path not set.")

        return Tokenizer.from_file(self.tokeniser_path)

    def test_tokeniser(self, tokeniser: Tokenizer, sample = "Also sprach Zarathustra: the eternal recurrence of the same."):
        loaded_tokeniser = tokeniser
        encoding = loaded_tokeniser.encode(sample)

        print(f"{len(sample)} characters -> {len(encoding.ids)} tokens\n")
        print("ids:   ", encoding.ids)
        print("pieces:", encoding.tokens)
        print("\ndecoded round-trip:", loaded_tokeniser.decode(encoding.ids))
        print("round-trip matches original:", loaded_tokeniser.decode(encoding.ids) == sample)
        