# Training.py
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")  # 無畫面環境也能存檔
import matplotlib.pyplot as plt
from datetime import datetime
import random

# ------------------------------------------------------------
# 1) 參數
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("CNN Training")
    p.add_argument("--stacked", type=str, default="stacked_data.npy",
                   help="經 Dataset.py 前處理完的 NPY（形狀 ~ [N, 2, K]）")
    p.add_argument("--epochs", type=int, default=200, help="訓練週期數")
    p.add_argument("--batch", type=int, default=256, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--K", type=int, default=51, help="子載波數 K（輸入最後一維）")
    p.add_argument("--val_ratio", type=float, default=0.1, help="驗證集比例")
    p.add_argument("--ylim", type=str, default="", 
                   help="Loss 圖 y 軸範圍，例如 '1.02,1.30'；留空自動縮放")
    p.add_argument("--out", type=str, default="cnn_basic.pt", help="最佳模型輸出檔名")
    return p.parse_args()

# ------------------------------------------------------------
# 2) 隨機種子（可重現）
# ------------------------------------------------------------
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# 3) Dataset（與 Dataset.py 一致的最小版）
#    - 期待 stacked_data.npy 形狀: [N, 2, K]
#    - X: [N, 1, 1, K]，Y: [N, K]
# ------------------------------------------------------------
class CsiPairDataset(Dataset):
    def __init__(self, stacked_path: str, K: int = 51):
        super().__init__()
        arr = np.load(stacked_path)  # 期望形狀 [N, 2, K]
        if arr.ndim != 3 or arr.shape[1] != 2 or arr.shape[2] != K:
            raise ValueError(
                f"[Dataset] 期望 shape=[N,2,{K}]，實際={arr.shape}，請檢查 stacked_data 檔案。"
            )
        # X = 第一通道（ex: Alice），Y = 第二通道（ex: Bob）
        x = arr[:, 0, :]    # [N, K]
        y = arr[:, 1, :]    # [N, K]

        # 轉為 torch tensor
        self.x = torch.from_numpy(x.astype(np.float32)).unsqueeze(1).unsqueeze(1)  # [N,1,1,K]
        self.y = torch.from_numpy(y.astype(np.float32))                            # [N,K]
        self.K = K

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ------------------------------------------------------------
# 4) 模型：嘗試從你的現有檔案載入；失敗則用 fallback
#    - 你的專案中通常有 CNN.py 裡的 class cnn_basic
# ------------------------------------------------------------
def build_model(K: int):
    # 嘗試 1：CNN.py
    try:
        from CNN import cnn_basic as Model
        print("[Model] 使用 CNN.py::cnn_basic")
        return Model()
    except Exception as e1:
        print(f"[Model] 載入 CNN.py 失敗，嘗試 cnn_basic.py :: {e1}")

    # 嘗試 2：cnn_basic.py
    try:
        from cnn_basic import cnn_basic as Model
        print("[Model] 使用 cnn_basic.py::cnn_basic")
        return Model()
    except Exception as e2:
        print(f"[Model] 載入 cnn_basic.py 失敗，改用 fallback :: {e2}")

    # Fallback 模型：Conv(1xK) + MLP -> K
    class FallbackCNN(nn.Module):
        def __init__(self, K):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=(1,3), padding=(0,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 1, kernel_size=(1,1)),
                nn.ReLU(inplace=True),
            )
            self.mlp = nn.Sequential(
                nn.Linear(K, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, K),
                nn.Sigmoid()
            )
        def forward(self, x):          # x: [B,1,1,K]
            z = self.conv(x)           # [B,1,1,K]
            z = z.flatten(start_dim=1) # [B,K]
            return self.mlp(z)         # [B,K]
    print("[Model] 使用 FallbackCNN")
    return FallbackCNN(K)

# ------------------------------------------------------------
# 5) 訓練／驗證
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

# ------------------------------------------------------------
# 6) 畫圖工具
# ------------------------------------------------------------
def save_loss_curves(train_hist, val_hist, ylim_pair=None, prefix="loss_curve"):
    # 全域圖
    plt.figure(figsize=(8,5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CNN Training Curve")
    plt.legend()
    if ylim_pair is not None:
        plt.ylim(ylim_pair[0], ylim_pair[1])
    plt.tight_layout()
    plt.savefig(f"{prefix}.png", dpi=160)
    plt.close()
    print(f" Saved: {prefix}.png")

    # 0~25 epoch 的 zoom-in
    cut = min(25, len(train_hist))
    plt.figure(figsize=(6,4))
    plt.plot(train_hist[:cut], label="Train")
    plt.plot(val_hist[:cut], label="Val")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Zoomed Loss Curve (0~25 Epochs)")
    plt.legend()
    if ylim_pair is not None:
        plt.ylim(ylim_pair[0], ylim_pair[1])
    plt.tight_layout()
    plt.savefig(f"{prefix}_zoom.png", dpi=160)
    plt.close()
    print(f" Saved: {prefix}_zoom.png")

# ------------------------------------------------------------
# 7) 主流程
# ------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(729)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device = {device}")

    # 解析 ylim
    ylim_pair = None
    if args.ylim:
        try:
            lo, hi = args.ylim.split(",")
            ylim_pair = (float(lo), float(hi))
            print(f"[Plot] 使用自訂 y-lim = {ylim_pair}")
        except Exception as e:
            print(f"[Plot] --ylim 解析失敗，改用自動範圍 :: {e}")

    # 資料集
    print(f"[Data] 載入 {args.stacked}")
    ds = CsiPairDataset(args.stacked, K=args.K)
    N = len(ds)
    n_val = max(1, int(N * args.val_ratio))
    n_tr  = N - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(123))
    print(f"[Data] total={N}, train={n_tr}, val={n_val}")

    # DataLoader
    # Windows 建議 num_workers=0；Linux/RPi 可調 >0
    dl_tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       pin_memory=True, num_workers=0)
    dl_va = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       pin_memory=True, num_workers=0)

    # 模型 / 優化 / Loss
    model = build_model(args.K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 訓練
    best_val = float("inf")
    tr_hist, va_hist = [], []

    print(f"[Run] epochs={args.epochs} batch={args.batch} lr={args.lr}")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, loss_fn, device)
        va_loss = evaluate(model, dl_va, loss_fn, device)
        tr_hist.append(tr_loss)
        va_hist.append(va_loss)

        tag = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.out)
            tag = f"  <-- saved best to {args.out}"
        print(f"[{ep:03d}/{args.epochs}] train={tr_loss:.6f}  val={va_loss:.6f}{tag}")

    # 畫圖
    save_loss_curves(tr_hist, va_hist, ylim_pair=ylim_pair, prefix="loss_curve")

    print(f"[Done] best_val={best_val:.6f} | model={args.out}")

if __name__ == "__main__":
    main()
