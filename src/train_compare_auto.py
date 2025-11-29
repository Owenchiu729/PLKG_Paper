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
    """Build sliding windows of length W."""
    N, F = a.shape
    M = N - W + 1
    xa = np.stack([a[i : i + W] for i in range(M)], axis=0)
    xb = np.stack([b[i : i + W] for i in range(M)], axis=0)
    return xa, xb


def train_one(model, train_loader, x_val, y_val, device, epochs, lr, log_path):
    """Train a single model."""
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

    # Save log
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,val_mse\n")
        for i, v in enumerate(history, start=1):
            f.write(f"{i},{v}\n")

    return history



def plot_curves(curves):
    """
    Log-Scale plot function.
    Shows all 4 models (including Encoder) clearly by using logarithmic scale.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 設定裁切點：只避開前 5% 的垂直下降，保留大部分過程
    # 既然用了 Log Scale，我們可以看到更早期的收斂過程也不會壞掉
    hist_list = list(curves.values())
    max_epoch = max(len(h) for h in hist_list)
    start_idx = int(max_epoch * 0.05) 
    if start_idx < 1: start_idx = 0

    plt.figure(figsize=(8, 6), dpi=150)

    # 2. 畫出所有四條線 (不進行過濾)
    for name, hist in curves.items():
        # 繪製時數據對齊 X 軸
        # 注意：如果 hist 是 list，要確保長度對齊
        x_axis = range(1, len(hist) + 1)
        
        # 為了讓圖表更清爽，我們只畫 start_idx 之後的數據，
        # 但 X 軸座標要保持正確 (例如從第 10 輪畫到第 200 輪)
        if len(hist) > start_idx:
            plt.plot(
                x_axis[start_idx:], 
                hist[start_idx:], 
                label=name, 
                linewidth=2
            )

    # 3. 【關鍵修改】啟用對數座標
    plt.yscale('log')

    # 設定 X 軸範圍
    plt.xlim(start_idx, max_epoch)
    
    # 加上更細緻的格線 (Log scale 需要 'both' 才能看清刻度)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    
    plt.xlabel("Epochs")
    plt.ylabel("MSE (Log Scale)")
    plt.title("Model Comparison (Log Scale)")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("compare_log.png")
    print(f"[DONE] Saved compare_log.png")


# ---------------- Main ----------------
def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to normalized_training_set.npy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    # ----- Load CSI data -----
    data = np.load(args.data)
    A = data[0]
    B = data[1]

    mu = A.mean()
    std = A.std() + 1e-8
    A = (A - mu) / std
    B = (B - mu) / std

    # ----- Build windows for temporal pooling -----
    W = args.window
    Aw, Bw = make_windows(A, B, W)
    mid = W // 2

    # Single-frame view
    A1 = Aw[:, mid, :]
    B1 = Bw[:, mid, :]

    # Shuffle and split train/val
    # 不要打亂 (Shuffle)，直接按時間順序切分，避免資料洩漏
    M = A1.shape[0]
    split = int(M * 0.8) # 前 80% 做訓練，後 20% 做驗證

    # 準備數據轉換函數
    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    F = A1.shape[1]

    # --- Single-frame tensors (直接切分) ---
    # 訓練集：前 80%
    xtr1 = to_tensor(A1[:split])
    ytr1 = to_tensor(B1[:split])
    # 驗證集：後 20%
    xva1 = to_tensor(A1[split:]).to(device)
    yva1 = to_tensor(B1[split:]).to(device)

    # DataLoader 裡面的 shuffle=True 是可以的，因為只在訓練集內部打亂
    train_loader_1 = DataLoader(
        TensorDataset(xtr1, ytr1),
        batch_size=args.batch,
        shuffle=True, 
    )

    # --- Window tensors (直接切分) ---
    xtrw = to_tensor(Aw[:split])
    ytrw = to_tensor(Bw[:split])
    xvaw = to_tensor(Aw[split:]).to(device)
    yvaw = to_tensor(Bw[split:]).to(device)

    # 【修正點 1】移除重複定義，並確保括號閉合
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
        enc, train_loader_1, xva1, yva1, device, args.epochs, args.lr, "logs/encoder.csv"
    )

    print("[FNN] training...")
    # 【修正點 2】補回 FNN 的訓練邏輯
    fnn = FNN(F).to(device)
    curves["FNN"] = train_one(
        fnn, train_loader_1, xva1, yva1, device, args.epochs, args.lr, "logs/fnn.csv"
    )

    print("[CNN] training...")
    cnn = CNN(F).to(device)
    curves["CNN"] = train_one(
        cnn, train_loader_1, xva1, yva1, device, args.epochs, args.lr, "logs/cnn.csv"
    )

    print("[CNN-TP] training...")
    cnntp = CNN_TP(F, W).to(device)
    curves["CNN-TP"] = train_one(
        cnntp, train_loader_w, xvaw, yvaw, device, args.epochs, args.lr, "logs/cnntp.csv"
    )
    
    return curves


# Entry point
if __name__ == "__main__":
    curves = main()
    plot_curves(curves)