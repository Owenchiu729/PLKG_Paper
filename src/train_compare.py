import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt


# ---------------- Models ----------------


class Encoder(nn.Module):
    """Shallow encoder baseline (1-layer MLP)."""

    def __init__(self, F: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, F),
            nn.LayerNorm(F),
            nn.ReLU(),
            nn.Linear(F, F),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        return self.net(x)


class FNN(nn.Module):
    """Two-layer MLP baseline."""

    def __init__(self, F: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, F),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        return self.net(x)


class CNN(nn.Module):
    """1D CNN + small MLP on single CSI frame."""

    def __init__(self, F: int) -> None:
        super().__init__()
        # Treat input as [B, 1, 1, F]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.LayerNorm([4, 1, F]),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=(1, 1)),
            nn.LayerNorm([1, 1, F]),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(F, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, F),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        x = x.unsqueeze(1).unsqueeze(1)          # [B, 1, 1, F]
        h = self.conv(x).flatten(start_dim=1)    # [B, F]
        return self.head(h)


class CNN_TP(nn.Module):
    """CNN with temporal pooling window W on CSI sequence."""

    def __init__(self, F: int, W: int) -> None:
        super().__init__()
        self.W = W
        # Input: [B, 1, W, F]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(W, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(1, 1)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(F, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, F),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, F]
        x = x.unsqueeze(1)                       # [B, 1, W, F]
        h = self.conv(x)                         # [B, 1, 1, F]
        h = h.flatten(start_dim=1)               # [B, F]
        return self.head(h)


# ---------------- Utilities ----------------


def make_windows(a: np.ndarray, b: np.ndarray, W: int):
    """
    Build sliding windows of length W for temporal pooling.

    Input:
        a, b: [N, F]
    Output:
        xa, xb: [M, W, F], where M = N - W + 1
    """
    N, F = a.shape
    M = N - W + 1
    xa = np.stack([a[i : i + W] for i in range(M)], axis=0)
    xb = np.stack([b[i : i + W] for i in range(M)], axis=0)
    return xa, xb


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str,
    epochs: int,
    lr: float,
    log_path: str,
):
    """
    Train a single model with reciprocity-consistency MSE loss.

    loss = MSE( f(x_ap), f(x_sta) )
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = []

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred_a = model(xb)
            pred_b = model(yb)
            loss = mse(pred_a, pred_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(x_val), model(y_val)).item()
        history.append(val_loss)

        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep + 1:3d} | Val MSE {val_loss:.6e}")

    # Save log as CSV
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,val_mse\n")
        for i, v in enumerate(history, start=1):
            f.write(f"{i},{v}\n")

    return history


# ---------------- Main ----------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to normalized_training_set.npy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    # ----- Load CSI data -----
    # Expected shape: [2, N, F]  (AP, STA)
    data = np.load(args.data)
    A = data[0]  # [N, F]
    B = data[1]  # [N, F]

    # Normalize using AP statistics (same as in thesis code style)
    mu = A.mean()
    std = A.std() + 1e-8
    A = (A - mu) / std
    B = (B - mu) / std

    # ----- Build windows for temporal pooling -----
    W = args.window
    Aw, Bw = make_windows(A, B, W)  # [M, W, F]
    mid = W // 2

    # Single-frame view (center frame of each window)
    A1 = Aw[:, mid, :]  # [M, F]
    B1 = Bw[:, mid, :]  # [M, F]

    # Shuffle and split train/val
    M = A1.shape[0]
    idx = np.random.permutation(M)
    A1 = A1[idx]
    B1 = B1[idx]
    Aw = Aw[idx]
    Bw = Bw[idx]

    split = int(M * 0.8)

    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    F = A1.shape[1]

    # Single-frame tensors (Encoder / FNN / CNN)
    xtr1 = to_tensor(A1[:split])
    ytr1 = to_tensor(B1[:split])
    xva1 = to_tensor(A1[split:]).to(device)
    yva1 = to_tensor(B1[split:]).to(device)

    train_loader_1 = DataLoader(
        TensorDataset(xtr1, ytr1),
        batch_size=args.batch,
        shuffle=True,
    )

    # Window tensors (CNN-TP)
    xtrw = to_tensor(Aw[:split])
    ytrw = to_tensor(Bw[:split])
    xvaw = to_tensor(Aw[split:]).to(device)
    yvaw = to_tensor(Bw[split:]).to(device)

    train_loader_w = DataLoader(
        TensorDataset(xtrw, ytrw),
        batch_size=args.batch,
        shuffle=True,
    )

    curves = {}

    # ----- Train each model -----
    print("[Encoder] training...")
    enc = Encoder(F).to(device)
    curves["Encoder"] = train_one(
        enc,
        train_loader_1,
        xva1,
        yva1,
        device,
        args.epochs,
        args.lr,
        "logs/encoder.csv",
    )

    print("[FNN] training...")
    fnn = FNN(F).to(device)
    curves["FNN"] = train_one(
        fnn,
        train_loader_1,
        xva1,
        yva1,
        device,
        args.epochs,
        args.lr,
        "logs/fnn.csv",
    )

    print("[CNN] training...")
    cnn = CNN(F).to(device)
    curves["CNN"] = train_one(
        cnn,
        train_loader_1,
        xva1,
        yva1,
        device,
        args.epochs,
        args.lr,
        "logs/cnn.csv",
    )

    print("[CNN-TP] training...")
    cnntp = CNN_TP(F, W).to(device)
    curves["CNN-TP"] = train_one(
        cnntp,
        train_loader_w,
        xvaw,
        yvaw,
        device,
        args.epochs,
        args.lr,
        "logs/cnntp.csv",
    )
    return curves

def plot_curves (curves):

    # ============================
    #   Plot 1 — Full MSE curves
    # ============================
    plt.figure(figsize=(6, 4), dpi=150)

    for name, hist in curves.items():
        plt.plot(
            range(1, len(hist) + 1),
            np.asarray(hist),     # raw MSE
            label=name,
            linewidth=2,
        )

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Model Comparison (Full MSE Curve)")
    plt.grid(True, alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.ticklabel_format(axis="y", style="plain")   # prevent scientific auto-scaling
    plt.tight_layout()
    plt.savefig("compare_full.png")
    print("[DONE] Saved compare_full.png")


    # ==========================================
    #   Plot 2 — Zoomed MSE (1.5e-7 to 5e-7)
    # ==========================================
    plt.figure(figsize=(6, 4), dpi=150)

    for name, hist in curves.items():
         plt.plot(
        range(1, len(hist) + 1),
        np.asarray(hist),
        label=name,
        linewidth=2,
    )

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Model Comparison (Zoomed MSE)")
    plt.xlim(0, 150)


    plt.grid(True, alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.ticklabel_format(axis="y", style="sci")
    plt.tight_layout()
    plt.savefig("compare_zoom.png")
    print("[DONE] Saved compare_zoom.png")


    # ==========================================
    #   Plot 3 — Individual subplots per model
    # ==========================================
    model_names = ["Encoder", "FNN", "CNN", "CNN-TP"]

    plt.figure(figsize=(10, 6), dpi=150)

    for i, name in enumerate(model_names):
        plt.subplot(2, 2, i + 1)

        hist = np.asarray(curves[name])
        plt.plot(range(1, len(hist) + 1), hist, color="C{}".format(i), linewidth=2)

        plt.title(name)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.5)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    plt.suptitle("Individual MSE Curves")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("compare_subplots.png")
    print("[DONE] Saved compare_subplots.png")


# Entry point
if __name__ == "__main__":
    curves = main ()
    plot_curves(curves)


