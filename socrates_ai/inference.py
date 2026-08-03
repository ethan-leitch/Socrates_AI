import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# This is the file that is copied into my website repo, only to be used as
# a output generation tool.


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
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
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)


class SocratesAI:
    """Simple class that is used to produce text outputs. Only uses the essential classes and
    model structure to save on file size, and uses only the saved model weigths."""

    def __init__(self, weights_path, tokeniser_path, device="cpu",
                 d_model=256, n_layers=4, n_heads=4, block_size=128):
        self.device = device
        self.block_size = block_size
        self.tokeniser = Tokenizer.from_file(tokeniser_path)

        self.model = MiniTransformer(
            vocab_size=self.tokeniser.get_vocab_size(),
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            block_size=block_size, dropout=0.1,  # dropout value is irrelevant here, model.eval() below disables it regardless
        ).to(device)

        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # turns off dropout, switches layernorm/batchnorm to inference behaviour

    @torch.no_grad()
    def talk(self, prompt, max_new_tokens=100, temperature=0.85, stop_at_eos=True):
        ids = self.tokeniser.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = self.tokeniser.token_to_id("<eos>") if stop_at_eos else None

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self.model(idx_cond)
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break

        return self.tokeniser.decode(idx[0].tolist())
