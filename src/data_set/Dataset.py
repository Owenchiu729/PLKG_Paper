# -*- coding: utf-8 -*-
import numpy as np
import torch 
from torch.utils.data import Dataset
import pandas as pd

# ==============================================================================
#  STEP 1: DATA LOADING AND FEATURE ENGINEERING FUNCTION
# ==============================================================================

def load_and_process_csi_data(file_path_0, file_path_1):
    """
    Loads and processes CSI data using numerically stable and memory-efficient methods.
    """
    
    def _parse_and_extract_features(file_path):
        """
        Helper function to parse a single CSV, extract features,
        and ensure all rows have consistent length.
        """
        df = pd.read_csv(file_path)
        df.dropna(subset=['payload'], inplace=True)
        df = df[df['payload'].str.startswith('serial_num:')]
        # --- 這整段邏輯現在被正確地縮排在函式內部 ---
        processed_data = []
        target_len = None  # 第一筆樣本長度（用來保持一致）
        drop_cnt = 0

        for payload_str in df['payload']:
            try:
                # 移除標頭與多餘符號
                clean_str = payload_str.replace('serial_num:,', '', 1).strip(',"')
                if not clean_str:
                    continue

                # 分割字串為整數
                parts = clean_str.split(',')
                numeric_values = [int(p) for p in parts]

                # RSSI 正確索引是 [1]
                rssi = numeric_values[1]

                # 取 CSI 原始資料（略過 serial_num, RSSI, noise 等前幾欄）
                csi_raw = np.array(numeric_values[9:], dtype=np.int32)
                if csi_raw.size < 4:
                    drop_cnt += 1
                    continue

                # 確保實虛成對（偶數個）
                if csi_raw.size % 2 != 0:
                    csi_raw = csi_raw[:-1]

                csi_pairs = csi_raw.reshape(-1, 2)
                real_f = csi_pairs[:, 0].astype(np.float32, copy=False)
                imag_f = csi_pairs[:, 1].astype(np.float32, copy=False)

                # 1) Magnitude（數值穩定）
                magnitude = np.hypot(real_f, imag_f)

                # 2) Phase（相位展開 + 去平均 / 去共同相位偏移）
                phase = np.unwrap(np.arctan2(imag_f, real_f))
                phase -= phase.mean(dtype=np.float64)

                # 合併 RSSI + CSI 特徵
                final_features = np.concatenate(([rssi], magnitude, phase))

                # --- 確保所有樣本同長度 ---
                if target_len is None:
                    target_len = final_features.size
                elif final_features.size != target_len:
                    # 長度不一致 → 截斷或跳過
                    if final_features.size > target_len:
                        final_features = final_features[:target_len]
                    else:
                        drop_cnt += 1
                        continue

                processed_data.append(final_features)

            except (ValueError, IndexError):
                drop_cnt += 1
                continue

        if drop_cnt:
            print(f"  [WARN] In {file_path}, dropped {drop_cnt} rows due to parsing errors or length mismatch.")
            
        return np.array(processed_data, dtype=np.float32)
        # --- 縮排修正結束 ---

    # --- Main processing flow ---
    # (這一段現在可以正確地呼叫上面定義好的 _parse_and_extract_features)
    print("Processing data for device 0...")
    data_0 = _parse_and_extract_features(file_path_0)
    print(f"  - Found {len(data_0)} samples with {data_0.shape[1] if len(data_0) > 0 else 0} features.")
    
    print("Processing data for device 1...")
    data_1 = _parse_and_extract_features(file_path_1)
    print(f"  - Found {len(data_1)} samples with {data_1.shape[1] if len(data_1) > 0 else 0} features.")

    if len(data_0) == 0 or len(data_1) == 0:
        raise ValueError("One or both data files resulted in zero valid samples. Check CSV content or parsing logic.")

    min_len = min(len(data_0), len(data_1))
    data_0 = data_0[:min_len]
    data_1 = data_1[:min_len]
    print(f"Aligned sample length to: {min_len}")

    min_features = min(data_0.shape[1], data_1.shape[1])
    data_0 = data_0[:, :min_features]
    data_1 = data_1[:, :min_features]
    print(f"Aligned feature count to: {min_features}")
    
    # --- Advanced Z-Score Normalization (Memory Efficient) ---
    EPS = 1e-6
    M = (data_0.shape[1] - 1) // 2 

    mag0 = data_0[:, 1:1+M];  pha0 = data_0[:, 1+M:1+2*M]
    mag1 = data_1[:, 1:1+M];  pha1 = data_1[:, 1+M:1+2*M]

    n_mag = mag0.size + mag1.size
    sum_mag  = mag0.sum(dtype=np.float64) + mag1.sum(dtype=np.float64)
    sum2_mag = (mag0**2).sum(dtype=np.float64) + (mag1**2).sum(dtype=np.float64)
    mu_mag = sum_mag / n_mag
    sigma_mag = np.sqrt(max(sum2_mag / n_mag - mu_mag**2, 0.0))

    n_pha = pha0.size + pha1.size
    sum_pha  = pha0.sum(dtype=np.float64) + pha1.sum(dtype=np.float64)
    sum2_pha = (pha0**2).sum(dtype=np.float64) + (pha1**2).sum(dtype=np.float64)
    mu_pha = sum_pha / n_pha
    sigma_pha = np.sqrt(max(sum2_pha / n_pha - mu_pha**2, 0.0))
    
    print(f"Calculated Mag stats: mu={mu_mag:.4f}, sigma={sigma_mag:.4f}")
    print(f"Calculated Pha stats: mu={mu_pha:.4f}, sigma={sigma_pha:.4f}")

    np.subtract(mag0, mu_mag, out=mag0);  np.divide(mag0, sigma_mag + EPS, out=mag0)
    np.subtract(mag1, mu_mag, out=mag1);  np.divide(mag1, sigma_mag + EPS, out=mag1)
    np.subtract(pha0, mu_pha, out=pha0);  np.divide(pha0, sigma_pha + EPS, out=pha0)
    np.subtract(pha1, mu_pha, out=pha1);  np.divide(pha1, sigma_pha + EPS, out=pha1)

    stacked_data = np.stack([data_0, data_1], axis=0)
    
    return stacked_data

# ==============================================================================
#  STEP 2: PYTORCH DATASET CLASSES
# ==============================================================================

class base_Dataset(Dataset):
    """Base class to handle the stacked (2, N, M) data structure."""
    def __init__(self, stacked_data: np.array):
        self.data_device_0 = torch.from_numpy(stacked_data[0, :, 1:]).float()
        self.labels_device_1 = torch.from_numpy(stacked_data[1, :, 1:]).float()
        self.rssi_device_0 = torch.from_numpy(stacked_data[0, :, 0]).float()
        self.rssi_device_1 = torch.from_numpy(stacked_data[1, :, 0]).float()
        self.length = stacked_data.shape[1]

    def __len__(self):
        return self.length

class PairedCsiDataset(base_Dataset):
    """Returns one timestamp from each device."""
    def __init__(self, stacked_data: np.array):
        super().__init__(stacked_data)
        
    def __getitem__(self, index):
        return self.data_device_0[index], self.labels_device_1[index]

class CsiCnnDataset(base_Dataset):
    """Uses a sliding window to create samples for CNNs."""
    def __init__(self, stacked_data: np.array, window_size=5):
        super().__init__(stacked_data)
        self.window_size = window_size
        self.length = super().__len__() - self.window_size + 1

    def __getitem__(self, index):
        data_window = self.data_device_0[index : index + self.window_size]
        label = self.labels_device_1[index + self.window_size - 1]
        return data_window.unsqueeze(0), label
    
    def __len__(self):
        return self.length

# ==============================================================================
#  STEP 3: EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    file_usb0 = 'csi_data_usb0_001.csv'
    file_usb1 = 'csi_data_usb1_001.csv'

    try:
        print("--- Starting Data Loading and Processing ---")
        processed_csi_data = load_and_process_csi_data(file_usb0, file_usb1)
        print("--- Data Processing Complete! ---")
        print(f"\nFinal stacked data shape: {processed_csi_data.shape}")
        
        print("\n--- Testing PairedCsiDataset ---")
        dataset = PairedCsiDataset(processed_csi_data)
        
        if len(dataset) > 0:
            print(f"Dataset length: {len(dataset)}")
            sample_data, sample_label = dataset[0]
            print(f"Sample data shape (Device 0 features): {sample_data.shape}")
            print(f"Sample label shape (Device 1 features): {sample_label.shape}")
        else:
            print("Dataset is empty. Check your CSV files.")

        # ==========================================================
        # Optional: Save processed numpy arrays
        # ==========================================================
        np.save("stacked_data.npy", processed_csi_data)
        print(" Saved processed data to 'stacked_data.npy' successfully.")
        
        np.save("usb0_processed.npy", processed_csi_data[0])
        np.save("usb1_processed.npy", processed_csi_data[1])
        print(" Saved usb0/usb1 processed data as separate .npy files.")

    except FileNotFoundError:
        print(f"\nERROR: Could not find '{file_usb0}' or '{file_usb1}'.")
        print("Please make sure the CSV files are in the same directory as the script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")