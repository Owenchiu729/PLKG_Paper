# -*- coding: utf-8 -*-
# Standalone Training.py: 只需 CNN.py + stacked_data.npy

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 兼容 CNN 類名：cnn_basic 或 CNN（兩者擇一存在即可）
try:
    from CNN import cnn_basic as Net
except ImportError:
    from CNN import CNN as Net


class CsiPairDataset(Dataset):
    """
    讀 stacked_data.npy -> 產生:
      x: [B, 1, 2, K]  (Alice 的 mag/pha)
      y: [B, K]        (Bob   的 magnitude)
    """
    def __init__(self, stacked_path: str, K: int = 51):
        data = np.load(stacked_path)              # [2, N, F], F = 1 + M + M
        assert data.ndim == 3 and data.shape[0] == 2, "stacked_data 應為 [2, N, F]"
        a, b = data[0], data[1]                   # Alice, Bob

        F = a.shape[1]
        M = (F - 1) // 2                          # 子載波數 (e.g., 62 when F=125)
        K = min(K, M)

        a_mag = a[:, 1:1+M]
        a_pha = a[:, 1+M:1+2*M]
        b_mag = b[:, 1:1+M]

        x = np.stack([a_mag[:, :K], a_pha[:, :K]], axis=1).astype(np.float32)  # [N, 2, K]
        y = b_mag[:, :K].astype(np.float32)                                    # [N, K]

        self.x = torch.from_numpy(x)  # [N, 2, K]
        self.y = torch.from_numpy(y)  # [N, K]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # 補上 channel 維度 1 -> [1, 2, K]；模型期望 [B,1,2,K]
        return self.x[idx].unsqueeze(0), self.y[idx]


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)     # [B, K]
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        running += loss_fn(model(x), y).item() * x.size(0)
    return running / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacked", default="stacked_data.npy")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=51, help="子載波數；若 CNN.py 改成 62，這裡設 62")
    ap.add_argument("--save", default="cnn_basic.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CsiPairDataset(args.stacked, K=args.K)

    n = len(ds)
    n_val = max(1, int(n * args.val_split))
    n_tr  = n - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, pin_memory=True)

    # 有些 CNN.py 建構子需要 K，有些不需要；兩種都支援
    try:
        model = Net(args.K).to(device)
    except TypeError:
        model = Net().to(device)

    loss_fn  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    tr_hist, va_hist = [], []

    print(f"Samples total={n}, train={n_tr}, val={n_val}, device={device}")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, tr_ld, optimizer, loss_fn, device)
        va = eval_one_epoch(model, va_ld, loss_fn, device)
        tr_hist.append(tr); va_hist.append(va)
        print(f"[{ep:03d}/{args.epochs}] train={tr:.6f}  val={va:.6f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), args.save)
            print(f"  ? saved best to {args.save} (val={best:.6f})")

    # 畫 loss 曲線
    plt.figure(figsize=(6,4))
    plt.plot(tr_hist, label="Train")
    plt.plot(va_hist, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend(); plt.tight_layout()
    plt.title("CNN Training Curve")
    plt.savefig("loss_curve.png")

    print("Done. Best val loss:", best)
    print("Saved:", args.save, "and loss_curve.png")


if __name__ == "__main__":
    main()
