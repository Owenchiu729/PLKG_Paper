import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
# --- 載入 torch (為了 base_Dataset) ---
import torch

# ===================================================================
# 步驟 1: 解析 CSV
# ===================================================================
def _parse_and_extract_features(file_path, rssi_feature=True, target_len=124):
    """
    從 CSV 檔案解析 payload，提取 RSSI、幅度和相位。
    """
    print(f"Parsing: {os.path.basename(file_path)} ...")
    df = pd.read_csv(file_path)

    # 過濾掉 'ts_ns' < 0 的無效行（例如 header）
    if 'ts_ns' in df.columns:
        df = df[df['ts_ns'] > 0]
    
    # 過濾掉不包含 'serial_num:' 的行
    df = df[df['payload'].str.contains('serial_num:', na=False)]

    all_features = []
    
    for payload in tqdm(df['payload']):
        parts = payload.split(',')
        try:
            # 找到 'serial_num:' 的索引
            try:
                header_idx = parts.index('serial_num:')
            except ValueError:
                continue # 找不到 'serial_num:'，跳過此行

            # 提取 RSSI
            # rssi 在 'serial_num:' 之後的第 3 個位置 (idx+3)
            # 例如: serial_num:,217,256,-18,-93,0,48,67,0...
            # 索引 [header_idx+3] 就是 -18
            if header_idx + 3 >= len(parts):
                continue # payload 太短
            
            rssi = int(parts[header_idx + 3])
            
            # 提取 CSI (幅度和相位)
            # CSI 資料在 'serial_num:' 之後的第 9 個位置 (idx+9) 開始
            csi_data_str = parts[header_idx + 9:]
            
            # 轉換為整數
            csi_data_int = [int(x) for x in csi_data_str]

            # 檢查長度是否為偶數 (mag, pha, mag, pha...)
            if len(csi_data_int) % 2 != 0:
                csi_data_int = csi_data_int[:-1] # 丟棄最後一個
            
            # 分離幅度和相位
            magnitudes = np.array(csi_data_int[0::2]) # 偶數索引
            phases = np.array(csi_data_int[1::2])     # 奇數索引
            
            # --- 相位處理 ---
            # 1. 相位展開 (Unwrap)
            phases = np.unwrap(phases)
            # 2. 移除共同相位偏移 (CPO removal)
            phases = phases - phases.mean()
            
            # 組合特徵
            if rssi_feature:
                features = np.concatenate([[rssi], magnitudes, phases])
            else:
                features = np.concatenate([magnitudes, phases])
            
            all_features.append(features)

        except (ValueError, IndexError) as e:
            # print(f"Skipping malformed row: {e}")
            pass # 略過解析失敗的行

    # --- 長度對齊 ---
    # 確保所有樣本特徵長度一致 (處理 ESP32 偶爾少傳幾個子載波的問題)
    if not all_features:
        return np.array([]) # 處理空檔案

    if target_len == 0:
        # 自動偵測 (不建議，但保留彈性)
        target_len = max(len(f) for f in all_features)

    processed_features = []
    for f in all_features:
        if len(f) == target_len:
            processed_features.append(f)
        elif len(f) > target_len:
            processed_features.append(f[:target_len]) # 截斷
        else:
            # 填充 (pad with zero)
            padding = np.zeros(target_len - len(f))
            processed_features.append(np.concatenate([f, padding]))

    return np.array(processed_features)

# ===================================================================
# 步驟 2: 對齊資料
# ===================================================================
def _align_data(data_0, data_1):
    """
    對齊兩個裝置的樣本數和特徵數。
    """
    print("Aligning data...")
    # 1. 對齊樣本數 (N)
    min_len = min(len(data_0), len(data_1))
    data_0 = data_0[:min_len]
    data_1 = data_1[:min_len]
    
    # 2. 對齊特徵數 (F)
    min_features = min(data_0.shape[1], data_1.shape[1])
    data_0 = data_0[:, :min_features]
    data_1 = data_1[:, :min_features]
    
    print(f"Aligned shape (N, F): ({min_len}, {min_features})")
    return data_0, data_1, min_len, min_features

# ===================================================================
# 步驟 3: 標準化 (Scale)
# vvvv 【這就是修改過的地方】 vvvv
# ===================================================================
def _calculate_stats_and_normalize(mag0, mag1, pha0, pha1):
    """
    (學長的建議) 計算 Min-Max 尺度並進行 0-1 歸一化。
    """
    # 1. 計算幅度的 Min/Max
    print("Calculating Min-Max scale for Magnitude...")
    min_m = min(np.min(mag0), np.min(mag1))
    max_m = max(np.max(mag0), np.max(mag1))
    range_m = max_m - min_m
    if range_m == 0: range_m = 1.0 # 避免除以零
    
    print(f"  Mag scale: min={min_m:.4f}, max={max_m:.4f}, range={range_m:.4f}")

    # 2. 計算相位的 Min/Max
    print("Calculating Min-Max scale for Phase...")
    min_p = min(np.min(pha0), np.min(pha1))
    max_p = max(np.max(pha0), np.max(pha1))
    range_p = max_p - min_p
    if range_p == 0: range_p = 1.0 # 避免除以零

    print(f"  Pha scale: min={min_p:.4f}, max={max_p:.4f}, range={range_p:.4f}")

    # 3. 執行 In-place (原地) 0-1 歸一化
    # (X - X_min) / (X_max - X_min)
    print("Applying normalization...")
    
    np.subtract(mag0, min_m, out=mag0)
    np.divide(mag0, range_m, out=mag0)
    
    np.subtract(mag1, min_m, out=mag1)
    np.divide(mag1, range_m, out=mag1)
    
    np.subtract(pha0, min_p, out=pha0)
    np.divide(pha0, range_p, out=pha0)
    
    np.subtract(pha1, min_p, out=pha1)
    np.divide(pha1, range_p, out=pha1)

    return mag0, mag1, pha0, pha1
# ^^^^ 【以上是修改過的地方】 ^^^^

# ===================================================================
# 步驟 4: 主函式
# ===================================================================
def load_and_process_csi_data(file_path_0, file_path_1, rssi_feature=True, target_len=124):
    """
    主函式：載入、解析、對齊、標準化並堆疊兩個裝置的 CSI 資料。
    """
    # 1. 解析並提取特徵
    data_0 = _parse_and_extract_features(file_path_0, rssi_feature, target_len)
    data_1 = _parse_and_extract_features(file_path_1, rssi_feature, target_len)
    
    if data_0.size == 0 or data_1.size == 0:
        raise ValueError("One or both input files resulted in zero valid samples.")

    # 2. 對齊資料
    data_0, data_1, N, F = _align_data(data_0, data_1)
    
    # 計算子載波數量 M (F = 1(RSSI) + M(mag) + M(pha))
    M = (F - 1) // 2
    if (F - 1) % 2 != 0:
        print(f"Warning: Features count F={F} is odd. Assuming M={M}.")

    # 3. 分離特徵以便進行標準化
    rssi_0 = data_0[:, 0]
    mag_0  = data_0[:, 1:1+M]
    pha_0  = data_0[:, 1+M:1+2*M]
    
    rssi_1 = data_1[:, 0]
    mag_1  = data_1[:, 1:1+M]
    pha_1  = data_1[:, 1+M:1+2*M]

    # 4. 標準化 (使用修改後的 Min-Max 函式)
    mag_0, mag_1, pha_0, pha_1 = _calculate_stats_and_normalize(mag_0, mag_1, pha_0, pha_1)
    
    # 5. 重組資料
    # 注意：RSSI 也用 Min-Max
    rssi_all = np.concatenate([rssi_0, rssi_1])
    min_r, max_r = np.min(rssi_all), np.max(rssi_all)
    range_r = max_r - min_r
    if range_r == 0: range_r = 1.0
    
    np.subtract(rssi_0, min_r, out=rssi_0)
    np.divide(rssi_0, range_r, out=rssi_0)
    
    np.subtract(rssi_1, min_r, out=rssi_1)
    np.divide(rssi_1, range_r, out=rssi_1)

    # 重組回 (N, F)
    data_0_norm = np.concatenate([rssi_0[:, None], mag_0, pha_0], axis=1)
    data_1_norm = np.concatenate([rssi_1[:, None], mag_1, pha_1], axis=1)

    # 6. 堆疊
    stacked_data = np.stack([data_0_norm, data_1_norm], axis=0) # Shape: [2, N, F]
    
    return stacked_data, data_0_norm, data_1_norm


# --- 以下是 Dataset 類別 (在 Training.py 中會被使用) ---
# 這些類別 *不是* 在執行此檔案時被使用，而是被 Training.py 匯入

class base_Dataset(object):
    """
    (父類別)
    讀取已處理的 stacked_data.npy，並分離 X (device_0) 和 Y (device_1)
    """
    def __init__(self, stacked_data_path="stacked_data.npy", K=51):
        stacked_data = np.load(stacked_data_path)
        assert stacked_data.ndim == 3 and stacked_data.shape[0] == 2
        
        data_0 = stacked_data[0] # [N, F]
        data_1 = stacked_data[1] # [N, F]
        
        # F = 1(RSSI) + M(mag) + M(pha)
        F = data_0.shape[1]
        M = (F - 1) // 2
        
        # 截取 K 個子載波
        K = min(K, M)
        
        # X = Device 0 (AP) 的 幅度 + 相位
        # Y = Device 1 (Client) 的 幅度
        self.data_device_0   = data_0[:, 1:1+2*M] # [N, 2*M]
        self.labels_device_1 = data_1[:, 1:1+M]   # [N, M]

        # 只取 K 個子載波
        self.data_device_0 = np.concatenate([
            self.data_device_0[:, :K],         # mag
            self.data_device_0[:, M:M+K]     # pha
        ], axis=1).astype(np.float32)
        
        self.labels_device_1 = self.labels_device_1[:, :K].astype(np.float32)

        self.data_device_0 = torch.from_numpy(self.data_device_0)
        self.labels_device_1 = torch.from_numpy(self.labels_device_1)

        self.length = len(self.data_device_0)


class PairedCsiDataset(base_Dataset):
    """
    一對一配對 (X[i], Y[i])
    """
    def __init__(self, stacked_data_path="stacked_data.npy", K=51):
        super().__init__(stacked_data_path, K)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data_device_0[index], self.labels_device_1[index]


class CsiCnn