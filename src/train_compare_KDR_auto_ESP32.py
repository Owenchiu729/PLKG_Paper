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
        # 修改：移除 LayerNorm 以避免數值過度集中導致的模態崩塌
        self.net = nn.Sequential(
            nn.Linear(F, F * 2), 
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
    plt.figure(figsize=(8, 5), dpi=150)
    names = list(kdr_results.keys())
    values = list(kdr_results.values())
    
    bars = plt.bar(names, values, color=['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    
    plt.title(f"Key Disagreement Rate ({bits}-bit Quantization)")
    plt.ylabel("KDR (Lower is Better)")
    # 自動調整 Y 軸
    top_limit = max(values) * 1.2 if max(values) > 0 else 0.1
    plt.ylim(0, top_limit)
    
    plt.grid(axis='y', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"[DONE] Saved {filename}")


def quantize_multibit(x, bits=3):
    """
    將浮點數量化為 2^bits 個等級。
    """
    # 根據之前的觀察，數值很小，所以範圍設窄一點 (-0.5 ~ 0.5)
    min_val, max_val = -2.0, 2.0
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
        print(f"  [Debug] Model Output Distribution (First 1000 keys):")
        
        # 顯示前 10 個不重複值就好，避免太多
        display_keys = unique_keys[:15]
        suffix = "..." if len(unique_keys) > 15 else ""
        print(f"    Unique Values ({len(unique_keys)} total): {display_keys} {suffix}")
        
        # 如果只有極少數值，代表模型崩塌
        if len(unique_keys) < 3:
            print(f"    ⚠️ WARNING: Low Entropy! Keys are static (Bad Security).")
    # ===================
            
    return error_vals / total_vals if total_vals > 0 else 0.0


# ---------------- Main ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to normalized_training_set.npy")
    
    # === [新增] 讓使用者輸入實驗名稱，自動加上檔名前綴 ===
    parser.add_argument("--name", type=str, default="experiment", help="Name prefix for all output files")
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=5)
    
    args = parser.parse_args()

    # === [新增] 自動產生所有輸出檔名 ===
    fn_loss  = f"{args.name}_loss.png"
    fn_kdr   = f"{args.name}_kdr.png"
    fn_model = f"{args.name}_model.pth"
    fn_keys  = f"{args.name}_keys.txt"
    
    print(f"Output filenames will start with: '{args.name}'")

    # 1. Load Data
    data = np.load(args.data)
    print(f"Data shape detected: {data.shape}")
    if data.shape[0] == 2:
        A, B = data[0], data[1]
    else:
        A, B = data[:, 0, :], data[:, 1, :]
    print(f"Split into A: {A.shape}, B: {B.shape}")


    # Normalize
    mu, std = A.mean(), A.std() + 1e-8
    A = (A - mu) / std
    B = (B - mu) / std

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

    # 4. Plot MSE Curves (Use generated filename)
    plot_curves(curves, filename=fn_loss)

    # 5. Calculate KDR (5-bit)
    # 設定測試位元數：5-bit (32階)，讓誤差更容易顯現出來
    TEST_BITS = 7
    print(f"\n=== Calculating Multi-bit KDR ({TEST_BITS}-bit / {2**TEST_BITS}-levels) ===")
    kdr_results = {}
    
    # Raw Data KDR (Baseline) - 也要用同樣的 bits
    def np_quantize(x, bits=3):
        min_val, max_val = -2.0, 2.0 # 配合程式碼中的範圍
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
    
    # Plot KDR (Use generated filename)
    plot_kdr(kdr_results, filename=fn_kdr, bits=TEST_BITS)
    
    # === [新增] 儲存表現最好的模型 (CNN-TP) (Use generated filename) ===
    print(f"\n=== Saving Best Model to {fn_model} ===")
    torch.save(cnntp.state_dict(), fn_model)
    print(f"[Saved] Model saved to '{fn_model}'")

    # === [新增] 實際生成並儲存金鑰 (使用 CNN-TP) ===
    cnntp.eval()
    all_keys = []
    with torch.no_grad():
        # 這裡我們只用驗證集 (val_loader_w) 來示範，實際應用可用全部資料
        for xb, yb in val_loader_w:
            xb = xb.to(device)
            feat = cnntp(xb) # 取得特徵
            
            # 進行量化 (變成整數索引 0 ~ 255)
            keys = quantize_multibit(feat, bits=TEST_BITS)
            all_keys.append(keys.cpu().numpy())

    # 轉成一大串數字
    final_keys = np.concatenate(all_keys, axis=0).flatten()
    
    # 存成文字檔 (Use generated filename)
    np.savetxt(fn_keys, final_keys, fmt="%d", delimiter=",")
    
    # 或是轉成二進位字串存檔 (例如: 00001100...)
    # 這裡示範把前 100 個整數轉成二進位印出來看
    print("\n[Preview] Generated Keys (First 5 values in Binary):")
    for val in final_keys[:5]:
        # 轉成 8-bit 二進位格式
        print(f"  Int: {val:3d} -> Bin: {format(val, '08b')}")

    print(f"\n[Done] All keys saved to '{fn_keys}'. Total keys: {len(final_keys)}")

if __name__ == "__main__":
    main()