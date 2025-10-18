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
        """Helper function to parse a single CSV and extract features."""
        df = pd.read_csv(file_path)
        df.dropna(subset=['payload'], inplace=True)
        df = df[df['payload'].str.startswith('serial_num:')]
        
        processed_data = []
        for payload_str in df['payload']:
            try:
                clean_str = payload_str.replace('serial_num:,', '', 1).strip(',"')
                if not clean_str: continue
                
                parts = clean_str.split(',')
                numeric_values = [int(p) for p in parts]
                
                # Using index 3 for RSSI as previously determined
                rssi = numeric_values[3] 
                csi_raw = np.array(numeric_values[9:])

                if len(csi_raw) % 2 != 0:
                    csi_raw = csi_raw[:-1]
                
                csi_pairs = csi_raw.reshape(-1, 2)
                real_part = csi_pairs[:, 0]
                imag_part = csi_pairs[:, 1]
                
                # --- Advanced Feature Calculation ---
                # 1. Magnitude (numerically stable)
                real_f = real_part.astype(np.float32, copy=False)
                imag_f = imag_part.astype(np.float32, copy=False)
                magnitude = np.hypot(real_f, imag_f)

                # 2. Phase + CPO removal
                phase = np.unwrap(np.arctan2(imag_f, real_f))
                phase -= phase.mean(dtype=np.float64)
                
                final_features = np.concatenate(([rssi], magnitude, phase))
                processed_data.append(final_features)

            except (ValueError, IndexError):
                continue
        
        return np.array(processed_data, dtype=np.float32)

    # --- Main processing flow ---
    print("Processing data for device 0...")
    data_0 = _parse_and_extract_features(file_path_0)
    print(f"  - Found {len(data_0)} samples with {data_0.shape[1]} features.")
    
    print("Processing data for device 1...")
    data_1 = _parse_and_extract_features(file_path_1)
    print(f"  - Found {len(data_1)} samples with {data_1.shape[1]} features.")

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
    # Number of subcarriers (M) for mag/pha features
    M = (data_0.shape[1] - 1) // 2 

    # Extract segments for in-place modification
    mag0 = data_0[:, 1:1+M];  pha0 = data_0[:, 1+M:1+2*M]
    mag1 = data_1[:, 1:1+M];  pha1 = data_1[:, 1+M:1+2*M]

    # Calculate statistics for Magnitude
    n_mag = mag0.size + mag1.size
    sum_mag  = mag0.sum(dtype=np.float64) + mag1.sum(dtype=np.float64)
    sum2_mag = (mag0**2).sum(dtype=np.float64) + (mag1**2).sum(dtype=np.float64)
    mu_mag = sum_mag / n_mag
    sigma_mag = np.sqrt(max(sum2_mag / n_mag - mu_mag**2, 0.0))

    # Calculate statistics for Phase
    n_pha = pha0.size + pha1.size
    sum_pha  = pha0.sum(dtype=np.float64) + pha1.sum(dtype=np.float64)
    sum2_pha = (pha0**2).sum(dtype=np.float64) + (pha1**2).sum(dtype=np.float64)
    mu_pha = sum_pha / n_pha
    sigma_pha = np.sqrt(max(sum2_pha / n_pha - mu_pha**2, 0.0))
    
    print(f"Calculated Mag stats: mu={mu_mag:.4f}, sigma={sigma_mag:.4f}")
    print(f"Calculated Pha stats: mu={mu_pha:.4f}, sigma={sigma_pha:.4f}")

    # Apply normalization in-place
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

    except FileNotFoundError:
        print(f"\nERROR: Could not find '{file_usb0}' or '{file_usb1}'.")
        print("Please make sure the CSV files are in the same directory as the script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")