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
    def __init__(self, F: int) -> None:
        super().__init__()
        # 移除 LayerNorm 以避免數值過度集中導致的模態崩塌
        self.net = nn.Sequential(
            nn.Linear(F, F * 2), 
            nn.BatchNorm1d(F * 2), # 建議：加上 BatchNorm 稍微穩定訓練 (非必須，但對 Speed 數據有幫助)
            nn.ReLU(),
            nn.Linear(F * 2, F),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FNN(nn.Module):
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
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, F: int) -> None:
        super().__init__()
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
        x = x.unsqueeze(1).unsqueeze(1)
        h = self.conv(x).flatten(start_dim=1)
        return self.head(h)


class CNN_TP(nn.Module):
    def __init__(self, F: int, W: int) -> None:
        super().__init__()
        self.W = W
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
        x = x.unsqueeze(1)
        h = self.conv(x).flatten(start_dim=1)
        return self.head(h)


# ---------------- Utilities ----------------


def make_windows(a: np.ndarray, b: np.ndarray, W: int):
    N, F = a.shape
    M = N - W + 1
    xa = np.stack([a[i : i + W] for i in range(M)], axis=0)
    xb = np.stack([b[i : i + W] for i in range(M)], axis=0)
    return xa, xb


def train_one(model, train_loader, x_val, y_val, device, epochs, lr, log_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = []

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # BN 需要 batch > 1
            if xb.shape[0] < 2: continue 
            
            loss = mse(model(xb), model(yb))
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(x_val), model(y_val)).item()
        history.append(val_loss)
        
        # 減少 log 頻率
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep + 1:3d} | Val MSE {val_loss:.6e}")

    log_dir = os.path.dirname(log_path)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,val_mse\n")
        for i, v in enumerate(history, start=1):
            f.write(f"{i},{v}\n")
    return history


def plot_curves(curves, filename):
    hist_list = list(curves.values())
    max_epoch = max(len(h) for h in hist_list)
    start_idx = int(max_epoch * 0.05)
    if start_idx < 1: start_idx = 0

    plt.figure(figsize=(8, 6), dpi=150)
    for name, hist in curves.items():
        if len(hist) > start_idx:
            plt.plot(range(start_idx+1, len(hist)+1), hist[start_idx:], label=name, linewidth=2)

    plt.yscale('log')
    plt.xlim(start_idx, max_epoch)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel("Epochs")
    plt.ylabel("MSE (Log Scale)")
    plt.title("Model Comparison (Log Scale)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[DONE] Saved {filename}")


def plot_kdr(kdr_results, filename, bits):
    # === 修改：使用 Log Scale 畫 KDR ===
    plt.figure(figsize=(8, 6), dpi=150)
    names = list(kdr_results.keys())
    values = list(kdr_results.values())
    
    # 處理 0 值以便在 Log Scale 顯示 (設為極小值)
    plot_values = [v if v > 0 else 1e-6 for v in values]
    
    bars = plt.bar(names, plot_values, color=['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    
    plt.yscale('log') # 開啟 Log Scale
    plt.ylim(1e-5, 1.5) # 設定範圍
    
    plt.title(f"Key Disagreement Rate ({bits}-bit Quantization)")
    plt.ylabel("KDR (Log Scale)")
    plt.grid(axis='y', alpha=0.5, which='both')
    
    for i, bar in enumerate(bars):
        real_val = values[i]
        # 文字位置調整
        y_pos = bar.get_height()
        if y_pos < 1e-4: y_pos = 1e-4
        
        plt.text(bar.get_x() + bar.get_width()/2, y_pos * 1.3, 
                 f"{real_val:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"[DONE] Saved {filename}")


def quantize_multibit(x, bits=3):
    """
    將浮點數量化為 2^bits 個等級。
    """
    # === [關鍵修改 2] 放寬量化範圍，避免 Speed 數據的突波被切掉 ===
    min_val, max_val = -3.0, 3.0  # 原本是 -2.0, 2.0
    
    levels = 2 ** bits
    step = (max_val - min_val) / levels
    
    # Clamp and map to 0 ~ levels-1
    x_clamped = torch.clamp(x, min_val, max_val - 1e-5)
    q_indices = ((x_clamped - min_val) / step).floor().long()
    return q_indices

def calculate_kdr(model, loader, device, bits=3):
    """Evaluate KDR using Multi-bit quantization with Entropy Check"""
    model.eval()
    total_vals = 0
    error_vals = 0
    
    # 用來檢查生成出來的金鑰長什麼樣子
    all_keys_a = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            feat_a = model(xb)
            feat_b = model(yb)
            
            # 使用多位元量化
            key_a = quantize_multibit(feat_a, bits=bits)
            key_b = quantize_multibit(feat_b, bits=bits)
            
            # 收集一部分 Key 供後續檢查
            if len(all_keys_a) < 1000:
                all_keys_a.extend(key_a.cpu().numpy().flatten())

            # 只要量化後的整數不一樣，就算錯
            diff = (key_a != key_b).float()
            
            error_vals += torch.sum(diff).item()
            total_vals += diff.numel()

    # === Debug 診斷區 ===
    if len(all_keys_a) > 0:
        unique_keys, counts = np.unique(all_keys_a, return_counts=True)
        # print(f"  [Debug] Model Output Distribution (First 1000 keys):")
        # display_keys = unique_keys[:15]
        # suffix = "..." if len(unique_keys) > 15 else ""
        # print(f"    Unique Values ({len(unique_keys)} total): {display_keys} {suffix}")
        
        if len(unique_keys) < 3:
            print(f"    ⚠️ WARNING: Low Entropy! Keys are static (Bad Security).")
    # ===================
            
    return error_vals / total_vals if total_vals > 0 else 0.0


# ---------------- Main ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to normalized_training_set.npy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=5)
    
    parser.add_argument("--out_loss", type=str, default="compare_log.png", help="Filename for loss curve plot")
    parser.add_argument("--out_kdr", type=str, default="compare_kdr.png", help="Filename for KDR plot")
    
    args = parser.parse_args()

    # 1. Load Data
    data = np.load(args.data)
    A, B = data[0], data[1]

    # === [關鍵修改 1] 獨立正規化 (Independent Normalization) ===
    # 為了解決 Alice 和 Bob 之間的 DC Offset，必須各自減去各自的平均值
    print("Applying Independent Normalization (Aligning Alice & Bob)...")
    mu_a, std_a = A.mean(), A.std() + 1e-8
    mu_b, std_b = B.mean(), B.std() + 1e-8
    
    A = (A - mu_a) / std_a
    B = (B - mu_b) / std_b
    # ========================================================

    # 2. Prepare Data (No Shuffle!)
    W = args.window
    Aw, Bw = make_windows(A, B, W)
    mid = W // 2
    A1, B1 = Aw[:, mid, :], Bw[:, mid, :]

    M = A1.shape[0]
    split = int(M * 0.8)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    F = A1.shape[1]
    def to_tensor(x): return torch.tensor(x, dtype=torch.float32)
    
    xtr1, ytr1 = to_tensor(A1[:split]), to_tensor(B1[:split])
    xva1, yva1 = to_tensor(A1[split:]).to(device), to_tensor(B1[split:]).to(device)
    
    train_loader_1 = DataLoader(TensorDataset(xtr1, ytr1), batch_size=args.batch, shuffle=True)
    val_loader_1 = DataLoader(TensorDataset(xva1.cpu(), yva1.cpu()), batch_size=args.batch, shuffle=False)

    xtrw, ytrw = to_tensor(Aw[:split]), to_tensor(Bw[:split])
    xvaw, yvaw = to_tensor(Aw[split:]).to(device), to_tensor(Bw[split:]).to(device)

    train_loader_w = DataLoader(TensorDataset(xtrw, ytrw), batch_size=args.batch, shuffle=True)
    val_loader_w = DataLoader(TensorDataset(xvaw.cpu(), yvaw.cpu()), batch_size=args.batch, shuffle=False)

    curves = {}

    # 3. Train Models
    print("\n=== Training Encoder ===")
    enc = Encoder(F).to(device)
    curves["Encoder"] = train_one(enc, train_loader_1, xva1, yva1, device, args.epochs, args.lr, "logs/encoder.csv")

    print("\n=== Training FNN ===")
    fnn = FNN(F).to(device)
    curves["FNN"] = train_one(fnn, train_loader_1, xva1, yva1, device, args.epochs, args.lr, "logs/fnn.csv")

    print("\n=== Training CNN ===")
    cnn = CNN(F).to(device)
    curves["CNN"] = train_one(cnn, train_loader_1, xva1, yva1, device, args.epochs, args.lr, "logs/cnn.csv")

    print("\n=== Training CNN-TP ===")
    cnntp = CNN_TP(F, W).to(device)
    curves["CNN-TP"] = train_one(cnntp, train_loader_w, xvaw, yvaw, device, args.epochs, args.lr, "logs/cnntp.csv")

    # 4. Plot MSE Curves
    plot_curves(curves, filename=args.out_loss)

    # 5. Calculate KDR
    # 建議：保持 8-bit 來展現模型的強大，或改為 7-bit 來獲得更漂亮的階梯圖
    TEST_BITS = 8 
    print(f"\n=== Calculating Multi-bit KDR ({TEST_BITS}-bit / {2**TEST_BITS}-levels) ===")
    kdr_results = {}
    
    # Raw Data KDR (Baseline)
    def np_quantize(x, bits=3):
        # === [關鍵修改 2 同步] ===
        min_val, max_val = -3.0, 3.0 
        levels = 2 ** bits
        step = (max_val - min_val) / levels
        x_clamped = np.clip(x, min_val, max_val - 1e-5)
        return np.floor((x_clamped - min_val) / step)

    raw_q_a = np_quantize(A1[split:], bits=TEST_BITS)
    raw_q_b = np_quantize(B1[split:], bits=TEST_BITS)
    raw_diff = np.sum(raw_q_a != raw_q_b)
    kdr_results["Raw Data"] = raw_diff / raw_q_a.size

    # Models KDR
    kdr_results["Encoder"] = calculate_kdr(enc, val_loader_1, device, bits=TEST_BITS)
    kdr_results["FNN"]     = calculate_kdr(fnn, val_loader_1, device, bits=TEST_BITS)
    kdr_results["CNN"]     = calculate_kdr(cnn, val_loader_1, device, bits=TEST_BITS)
    kdr_results["CNN-TP"]  = calculate_kdr(cnntp, val_loader_w, device, bits=TEST_BITS)

    print("KDR Results:", {k: f"{v:.4f}" for k, v in kdr_results.items()})
    
    # Plot KDR
    plot_kdr(kdr_results, filename=args.out_kdr, bits=TEST_BITS)

if __name__ == "__main__":
    main()