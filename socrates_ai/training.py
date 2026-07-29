import glob
import math
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class TrainingSocrates:
    def __init__(
        self,
        model,
        train_ids,
        val_ids,
        block_size=128,
        batch_size=64,
        device="cpu",
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=200,
        decay_target_steps=4000,
        grad_clip=1.0,
        checkpoint_dir=None,
    ):
        self.model = model.to(device)
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        # the step (counted globally, across every train()/resumed session,
        # not per call) by which decay reaches min_lr -- training can keep
        # going past this at min_lr, it just stops decaying further
        self.decay_target_steps = decay_target_steps
        self.grad_clip = grad_clip

        self.checkpoint_dir = checkpoint_dir or os.path.join("..", "data", "checkpoints")

        # bound to THIS model's parameters at construction time -- load_checkpoint()
        # restores this optimizer's state in place rather than replacing it,
        # so it can never end up pointing at a different model's tensors
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)

        self.history = []
        self.step = 0  # total steps trained so far, persists across train() calls / resumed sessions

    def get_batch(self, data):
        max_start = len(data) - self.block_size - 1
        start_idx = torch.randint(0, max_start, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in start_idx])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in start_idx])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self, data, eval_iters=15):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = self.get_batch(data)
            _, loss = self.model(x, y)
            losses[i] = loss.item()
        self.model.train()
        return losses.mean().item()

    def _lr_for_step(self, step):
        # keyed off the GLOBAL step count, not the size of whatever train()
        # call happens to be running -- warmup only ever happens once, in the
        # first warmup_steps of all training ever, and decay progresses
        # smoothly toward decay_target_steps no matter how many separate
        # sessions it takes to get there. A resumed session doesn't reset
        # either phase: the optimizer state (Adam's momentum) is restored
        # from the checkpoint too, so there's nothing unstable left for a
        # fresh warmup to protect against -- re-triggering it would only
        # waste steps re-ramping back up to where the LR already was
        if step < self.warmup_steps:
            return self.lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.decay_target_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.lr - self.min_lr) * cosine

    def train(self, max_steps=2000, eval_interval=200, eval_iters=15, verbose=True):
        start_time = time.time()

        for step_in_run in range(1, max_steps + 1):
            self.step += 1
            lr = self._lr_for_step(self.step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            x, y = self.get_batch(self.train_ids)
            logits, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # clip gradient norm before stepping -- guards against occasional
            # large gradient spikes destabilising training, standard practice
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if step_in_run % eval_interval == 0 or step_in_run == 1 or step_in_run == max_steps:
                train_loss = self.estimate_loss(self.train_ids, eval_iters)
                val_loss = self.estimate_loss(self.val_ids, eval_iters)
                self.history.append({
                    "step": self.step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                })
                if verbose:
                    print(f"step {self.step:6d} | lr {lr:.2e} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        elapsed = time.time() - start_time
        print(f"\nRun finished in {elapsed:.1f}s ({self.step} total steps trained so far)")

        # auto-save at the end of every run so progress is never lost to a
        # forgotten manual save -- call save_checkpoint() yourself any time
        # you want an extra save mid-run too
        self.save_checkpoint()

    def save_checkpoint(self, name=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        name = name or f"step_{self.step}.pt"
        path = os.path.join(self.checkpoint_dir, name)

        # architecture recorded alongside the weights specifically so
        # load_checkpoint() can catch a mismatched model with one clear
        # error, instead of either a cryptic raw PyTorch shape-mismatch
        # trace, or (worse) silently loading something that doesn't match
        architecture = {k: tuple(v.shape) for k, v in self.model.state_dict().items()}

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "step": self.step,
            "architecture": architecture,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path=None):
        if path is None:
            checkpoints = sorted(
                glob.glob(os.path.join(self.checkpoint_dir, "*.pt")),
                key=os.path.getmtime,
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            path = checkpoints[-1]

        checkpoint = torch.load(path, map_location=self.device)

        current_architecture = {k: tuple(v.shape) for k, v in self.model.state_dict().items()}
        if current_architecture != checkpoint["architecture"]:
            raise ValueError(
                f"Model architecture doesn't match checkpoint at {path}. "
                "The model passed into TrainingSocrates must have the exact "
                "same shape as when this checkpoint was saved -- check "
                "vocab_size/d_model/n_layers/n_heads/block_size haven't "
                "changed since."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.step = checkpoint["step"]
        print(f"Loaded checkpoint from {path} (resuming from step {self.step})")

    def plot_training_curves(self):
        if not self.history:
            print("No training history yet -- call train() first.")
            return

        steps = [h["step"] for h in self.history]
        train_losses = [h["train_loss"] for h in self.history]
        val_losses = [h["val_loss"] for h in self.history]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#1A1817")

        ax.plot(steps, train_losses, color="#18D5FFFF", alpha=0.95, linewidth=2, label="Train loss")
        ax.plot(steps, val_losses, color="#B02156FF", alpha=0.95, linewidth=2, label="Val loss")

        ax.grid(True, color="#444444", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#FF8818")

        ax.set_title("Training Loss Over Steps")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()

        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, eos_id=None):
        self.model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self.model(idx_cond)
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        self.model.train()
        return idx

    def talk(self, tokeniser, prompt, max_new_tokens=100, temperature=1.0, stop_at_eos=True):
        prompt_ids = tokeniser.encode(prompt).ids
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        eos_id = tokeniser.token_to_id("<eos>") if stop_at_eos else None
        out_ids = self.generate(idx, max_new_tokens, temperature, eos_id)
        return tokeniser.decode(out_ids[0].tolist())