import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 加上這行，避免在沒有螢幕的伺服器上報錯
import matplotlib.pyplot as plt

def inspect_data(npy_path, output_filename="data_inspection.png"):
    print(f"Loading {npy_path}...")
    try:
        data = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: 找不到檔案 {npy_path}")
        return

    # 假設 data shape 是 (2, N, F) -> (Alice/Bob, 樣本數, 子載波數)
    A = data[0]
    B = data[1]
    
    # 取第 0 個子載波的前 200 個時間點來看
    subcarrier_idx = 0 
    time_steps = 200
    
    # 防止資料長度不足 200
    limit = min(time_steps, A.shape[0])
    
    sample_A = A[:limit, subcarrier_idx]
    sample_B = B[:limit, subcarrier_idx]
    
    # 畫圖
    plt.figure(figsize=(12, 5))
    plt.plot(sample_A, label='Alice (Raw)', alpha=0.8)
    plt.plot(sample_B, label='Bob (Raw)', alpha=0.8)
    
    plt.title(f"CSI Waveform Inspection (First {limit} packets, Subcarrier {subcarrier_idx})")
    plt.xlabel("Time (Packets)")
    plt.ylabel("CSI Amplitude (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 計算相關係數 (Correlation Coefficient)
    corr = np.corrcoef(sample_A, sample_B)[0, 1]
    print(f"Alice-Bob Correlation: {corr:.4f}")
    
    if corr > 0.9:
        print(">> 診斷: 高度相關。環境可能很乾淨 (LOS)，容易訓練。")
    elif corr > 0.5:
        print(">> 診斷: 中度相關。典型的真實環境數據。")
    else:
        print(">> 診斷: 低相關。環境極度惡劣或是雜訊數據。")
        
    # 使用傳入的檔名存檔
    plt.savefig(output_filename)
    print(f"Saved inspection plot to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 設定參數
    parser.add_argument("--data", required=True, help="輸入的 .npy 檔案路徑")
    parser.add_argument("--out", type=str, default="data_inspection.png", help="輸出的圖片檔名")
    
    args = parser.parse_args()

    # 呼叫函式
    inspect_data(args.data, args.out)