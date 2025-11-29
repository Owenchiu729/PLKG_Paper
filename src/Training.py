import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- �s�W�Gmatplotlib �]�� Agg�A�L�����]��s�� ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CNN import cnn_basic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="stacked npy: shape (2,N,F)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, default="figs", help="where to save logs/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- reproducibility ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- load data ----
    data = np.load(args.data)  # shape: (2, N, F)
    ap, sta = data[0], data[1]
    N, F = ap.shape
    print(f"[INFO] Loaded {N} samples, {F} features")

    # ---- shuffle BEFORE split ----
    idx = np.random.permutation(N)
    ap, sta = ap[idx], sta[idx]

    # ---- train/test split ----
    train_size = int(N * 0.8)
    ap_train, sta_train = ap[:train_size], sta[:train_size]
    ap_test,  sta_test  = ap[train_size:], sta[train_size:]

    # ---- simple standardization (use AP stats to normalize both) ----
    x_mean, x_std = ap_train.mean(), ap_train.std() + 1e-8
    ap_train = (ap_train - x_mean) / x_std
    sta_train = (sta_train - x_mean) / x_std
    ap_test  = (ap_test  - x_mean) / x_std
    sta_test = (sta_test - x_mean) / x_std

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ap_train = torch.tensor(ap_train, dtype=torch.float32)
    sta_train = torch.tensor(sta_train, dtype=torch.float32)
    ap_test  = torch.tensor(ap_test,  dtype=torch.float32).to(device)
    sta_test = torch.tensor(sta_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(ap_train, sta_train),
                              batch_size=args.batch, shuffle=True)

    # ---- model / optim / loss ----
    model = cnn_basic(F).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    best = float("inf")
    patience, wait = 30, 0

    # ---- �s�W�Gloss ���v���� ----
    train_hist, val_hist = [], []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        nb = 0
        for apb, stab in train_loader:
            apb, stab = apb.to(device), stab.to(device)
            out_ap  = model(apb)
            out_sta = model(stab)
            loss = loss_fn(out_ap, out_sta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            nb += 1

        # per-epoch train loss�]�̫�]�i�Υ����^
        train_loss = running / max(nb, 1)

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            val_ap  = model(ap_test)
            val_sta = model(sta_test)
            val_loss = loss_fn(val_ap, val_sta).item()

        # ---- �����æC�L ----
        train_hist.append(float(train_loss))
        val_hist.append(float(val_loss))
        print(f"Epoch {epoch:3d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        # ---- early stop + save best ----
        if val_loss < best:
            best = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_key_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stop")
                break

    print(f"Best Val Loss: {best}")
    print("Saved key extractor -> best_key_model.pth")

    # ---- �s�W�G��X CSV �P �� ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"loss_log_{ts}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (tr, va) in enumerate(zip(train_hist, val_hist), start=1):
            f.write(f"{i},{tr},{va}\n")
    print(f"[INFO] Saved log -> {csv_path}")

    # Early(1~25) ���u
    early_n = min(25, len(train_hist))
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(range(1, early_n+1), train_hist[:early_n], label="Train")
    plt.plot(range(1, early_n+1), val_hist[:early_n],   label="Val")
    plt.title("Early Training (1�V25 epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.tight_layout()
    early_png = os.path.join(args.outdir, f"early_curve_{ts}.png")
    plt.savefig(early_png)

    # Full ���u
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(range(1, len(train_hist)+1), train_hist, label="Train")
    plt.plot(range(1, len(train_hist)+1), val_hist,   label="Val")
    plt.title("Full Training")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.tight_layout()
    full_png = os.path.join(args.outdir, f"full_curve_{ts}.png")
    plt.savefig(full_png)

    print(f"[INFO] Saved figs -> {early_png} , {full_png}")

if __name__ == "__main__":
    main()
