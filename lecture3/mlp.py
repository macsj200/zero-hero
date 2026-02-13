"""
MLP character-level language model from Karpathy's makemore lecture 3,
with Optuna hyperparameter search to beat the baseline dev loss of ~2.17.
"""

import random
import torch
import torch.nn.functional as F
import optuna


# --- Data ---

from pathlib import Path

_here = Path(__file__).resolve().parent
words = (_here.parent / "data" / "names.txt").read_text().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)  # 27

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
train_words = words[:n1]
dev_words = words[n1:n2]
test_words = words[n2:]


def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


# --- Training ---


def train_model(block_size, emb_dim, hidden_dim, n_hidden_layers, batch_size,
                lr_high, lr_low, steps, lr_switch_frac):
    Xtr, Ytr = build_dataset(train_words, block_size)
    Xdev, Ydev = build_dataset(dev_words, block_size)

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, emb_dim), generator=g)

    fan_in = block_size * emb_dim
    layers = []
    # first hidden layer
    W = torch.randn((fan_in, hidden_dim), generator=g) * (5 / 3) / (fan_in ** 0.5)
    b = torch.randn(hidden_dim, generator=g) * 0.01
    layers.append((W, b))
    # additional hidden layers
    for _ in range(n_hidden_layers - 1):
        W = torch.randn((hidden_dim, hidden_dim), generator=g) * (5 / 3) / (hidden_dim ** 0.5)
        b = torch.randn(hidden_dim, generator=g) * 0.01
        layers.append((W, b))
    # output layer
    W_out = torch.randn((hidden_dim, vocab_size), generator=g) * 0.01
    b_out = torch.randn(vocab_size, generator=g) * 0

    parameters = [C]
    for W, b in layers:
        parameters.extend([W, b])
    parameters.extend([W_out, b_out])
    for p in parameters:
        p.requires_grad = True

    switch_step = int(steps * lr_switch_frac)

    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        emb = C[Xtr[ix]]
        x = emb.view(-1, block_size * emb_dim)
        for W, b in layers:
            x = torch.tanh(x @ W + b)
        logits = x @ W_out + b_out
        loss = F.cross_entropy(logits, Ytr[ix])

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = lr_high if i < switch_step else lr_low
        for p in parameters:
            p.data += -lr * p.grad

    # evaluate dev loss
    emb = C[Xdev]
    x = emb.view(-1, block_size * emb_dim)
    for W, b in layers:
        x = torch.tanh(x @ W + b)
    logits = x @ W_out + b_out
    dev_loss = F.cross_entropy(logits, Ydev).item()

    # evaluate train loss
    emb = C[Xtr]
    x = emb.view(-1, block_size * emb_dim)
    for W, b in layers:
        x = torch.tanh(x @ W + b)
    logits = x @ W_out + b_out
    train_loss = F.cross_entropy(logits, Ytr).item()

    return train_loss, dev_loss


# --- Optuna ---


def objective(trial):
    block_size = trial.suggest_int("block_size", 3, 8)
    emb_dim = trial.suggest_int("emb_dim", 8, 30)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 400, step=50)
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    lr_high = trial.suggest_float("lr_high", 0.01, 0.5, log=True)
    lr_low = trial.suggest_float("lr_low", 0.001, 0.05, log=True)
    steps = trial.suggest_int("steps", 100_000, 300_000, step=50_000)
    lr_switch_frac = trial.suggest_float("lr_switch_frac", 0.3, 0.7)

    train_loss, dev_loss = train_model(
        block_size=block_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        batch_size=batch_size,
        lr_high=lr_high,
        lr_low=lr_low,
        steps=steps,
        lr_switch_frac=lr_switch_frac,
    )
    print(f"  train={train_loss:.4f}  dev={dev_loss:.4f}")
    return dev_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="makemore-mlp")
    study.optimize(objective, n_trials=50)

    print("\n=== Best Trial ===")
    print(f"Dev loss: {study.best_trial.value:.4f}")
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
