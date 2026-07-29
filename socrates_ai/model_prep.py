import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer



class Translator:
    def __init__(self, cleaned_books_path, Vocab_size=16000):
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

    def test_tokeniser(
        self,
        tokeniser: Tokenizer,
        sample="Also sprach Zarathustra: the eternal recurrence of the same.",
    ):
        loaded_tokeniser = tokeniser
        encoding = loaded_tokeniser.encode(sample)

        print(f"{len(sample)} characters -> {len(encoding.ids)} tokens\n")
        print("ids:   ", encoding.ids)
        print("pieces:", encoding.tokens)
        print("\ndecoded round-trip:", loaded_tokeniser.decode(encoding.ids))
        print(
            "round-trip matches original:",
            loaded_tokeniser.decode(encoding.ids) == sample,
        )


### Model architecture

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # one combined projection for Q, K and V instead of three separate
        # nn.Linear layers -- same result, fewer matmuls
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, d_model

        q, k, v = self.qkv_proj(x).split(C, dim=-1)

        # split d_model into n_heads smaller heads, each attending to the
        # sequence independently -- lets the model track several different
        # kinds of relationships at once (e.g. grammar vs. long-range topic)
        # instead of one head having to represent everything
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # is_causal=True masks out attention to future positions, so position t
        # can only look at positions <= t -- without this the model could
        # "cheat" during training by peeking at the very token it's supposed
        # to be predicting
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )

        # merge the heads back into one d_model-sized vector per position
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        # expand to 4x d_model in the hidden layer -- standard transformer
        # choice, gives the model room to compute something more complex than
        # a linear function of what attention produced, then project back
        # down to d_model so it can be added back onto the residual stream
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        # pre-norm + residual: x = x + sublayer(norm(x)), rather than
        # normalising after the sublayer. This is the modern standard
        # (GPT-2 onwards) because gradients can flow straight through the "+"
        # in every layer, all the way back to the input -- stacking many
        # layers with post-norm is known to be much harder to train stably
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        block_size=128,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # a transformer has no built-in sense of order the way a GRU does
        # (which processes tokens one at a time, in sequence) -- positional
        # embeddings are how the model learns "this token is 1st / 5th / 100th"
        # at all, since attention alone treats the input as an unordered set
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying: reuse the token embedding matrix (transposed) as the
        # output projection instead of learning a second (vocab_size x d_model)
        # matrix from scratch -- saves a meaningful fraction of parameters for
        # a small model, standard practice since GPT-2
        self.head.weight = self.token_emb.weight

        # see the markdown above -- without this, the tied weights inherit
        # nn.Embedding's default init scale, which is far too large for an
        # output projection and badly inflates the loss
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (B, T, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten batch and time together -- cross_entropy expects
            # (N, num_classes) predictions against (N,) targets, not a 3D tensor
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss
