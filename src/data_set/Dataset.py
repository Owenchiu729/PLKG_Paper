import numpy as np
import torch
from torch.utils.data import Dataset

class base_Dataset(Dataset):
    """Base Dataset class - Single ESP32 device"""
    def __init__(self, data: np.array):
        # data shape: (samples, features) - [RSSI, CSI_1, ..., CSI_51]
        self.rssi = torch.from_numpy(data[:, 0]).float()
        self.csi = torch.from_numpy(data[:, 1:52]).float()
    
    def __getitem__(self, index):
        # Return CSI and RSSI (or other labels)
        return self.csi[index], self.rssi[index]
    
    def __len__(self):
        return len(self.csi)


# Pure CSI - Basic CSI dataset
class csi_dataset(Dataset):
    def __init__(self, data: np.array):
        """
        data shape: (samples, 52) - [RSSI, CSI_1...CSI_51]
        For basic CSI feature learning
        """
        self.csi_data = torch.from_numpy(data[:, 1:52]).float()
        self.labels = torch.from_numpy(data[:, 0]).float()  # RSSI as label
    
    def __getitem__(self, index):
        return self.csi_data[index], self.labels[index]
    
    def __len__(self):
        return len(self.csi_data)


# CSI CNN approach - Using temporal window
class csi_cnn_dataset(Dataset):
    def __init__(self, data: np.array, window_size=2):
        """
        data shape: (samples, 52)
        Use sliding window to capture temporal correlation
        window_size: temporal window size
        """
        self.csi_data = torch.from_numpy(data[:, 1:52]).float()
        self.labels = torch.from_numpy(data[:, 0]).float()
        self.window_size = window_size
    
    def __getitem__(self, index):
        # Take window_size consecutive samples as input
        # shape: (1, window_size, 51) -> convert to (1, 1, window_size, 51)
        data = self.csi_data[index:index+self.window_size]  # (window_size, 51)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, 51)
        label = self.labels[index+self.window_size-1]  # Label of the last time step
        return data.squeeze(0), label  # Return (1, window_size, 51) for batch processing
    
    def __len__(self):
        return len(self.csi_data) - self.window_size + 1


# CSI CNN with LSTM - Time series with all features
class csi_cnn_lstm_dataset(Dataset):
    def __init__(self, data: np.array, window_size=2):
        """
        data shape: (samples, 52) - includes RSSI + CSI
        For LSTM time series prediction
        """
        self.all_data = torch.from_numpy(data).float()  # Include all features
        self.labels = torch.from_numpy(data[:, 0]).float()  # RSSI as label
        self.window_size = window_size
    
    def __getitem__(self, index):
        # shape: (1, window_size, 52) - includes all features
        data = self.all_data[index:index+self.window_size]  # (window_size, 52)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, 52)
        label = self.labels[index+self.window_size-1]
        return data.squeeze(0), label  # Return (1, window_size, 52)
    
    def __len__(self):
        return len(self.all_data) - self.window_size + 1


# Quantized CSI dataset - Quantized CSI (real + imaginary parts)
class csi_quan_dataset(Dataset):
    def __init__(self, data: np.array):
        """
        data shape: (samples, 103) - [RSSI, CSI_real_1...CSI_real_51, CSI_imag_1...CSI_imag_51]
        Quantized CSI contains real and imaginary parts (51*2=102)
        """
        self.csi_data = torch.from_numpy(data[:, 1:103]).float()  # 102-dim quantized CSI
        self.labels = torch.from_numpy(data[:, 0]).float()
    
    def __getitem__(self, index):
        return self.csi_data[index], self.labels[index]
    
    def __len__(self):
        return len(self.csi_data)


# CSI CNN quantization approach - Temporal window with quantized CSI
class csi_cnn_quan_dataset(Dataset):
    def __init__(self, data: np.array, window_size=2):
        """
        Quantized CSI + CNN temporal window
        """
        self.csi_data = torch.from_numpy(data[:, 1:103]).float()
        self.labels = torch.from_numpy(data[:, 0]).float()
        self.window_size = window_size
    
    def __getitem__(self, index):
        data = self.csi_data[index:index+self.window_size]  # (window_size, 102)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, 102)
        label = self.labels[index+self.window_size-1]
        return data.squeeze(0), label  # Return (1, window_size, 102)
    
    def __len__(self):
        return len(self.csi_data) - self.window_size + 1


# CSI CNN speed quantization - Quantized CSI + speed feature
class csi_cnn_speed_quan_dataset(Dataset):
    def __init__(self, data: np.array, window_size=2):
        """
        data shape: (samples, 104) - [RSSI, CSI_quan_102, speed]
        Includes quantized CSI and speed information
        """
        self.csi_data = torch.from_numpy(data[:, 1:104]).float()  # CSI + speed
        self.labels = torch.from_numpy(data[:, 1:103]).float()  # Only CSI as label
        self.window_size = window_size
    
    def __getitem__(self, index):
        data = self.csi_data[index:index+self.window_size]  # (window_size, 103)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, 103)
        label = self.labels[index+self.window_size-1]
        return data.squeeze(0), label  # Return (1, window_size, 103)
    
    def __len__(self):
        return len(self.csi_data) - self.window_size + 1


# Quantized CSI with LSTM - LSTM version with quantized CSI
class csi_cnn_quan_lstm_dataset(Dataset):
    def __init__(self, data: np.array, window_size=3):
        """
        Quantized CSI + LSTM (longer temporal window)
        """
        self.all_data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(data[:, 1:103]).float()  # Quantized CSI as label
        self.window_size = window_size
    
    def __getitem__(self, index):
        data = self.all_data[index:index+self.window_size]  # (window_size, 104 or all)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, features)
        label = self.labels[index+self.window_size-1]
        return data.squeeze(0), label  # Return (1, window_size, features)
    
    def __len__(self):
        return len(self.all_data) - self.window_size + 1


if __name__ == "__main__":
    # Testing: assume data format is (samples, 52)
    # Simulate ESP32 collected data
    dummy_data = np.random.randn(1000, 52)  # 1000 samples
    
    # Test basic CSI dataset
    test1 = csi_dataset(dummy_data)
    print("csi_dataset:")
    print("  Input shape:", test1[0][0].shape)   # (51,)
    print("  Label shape:", test1[0][1].shape)   # ()
    print("  Dataset length:", len(test1))
    
    # Test CNN dataset (temporal window)
    test2 = csi_cnn_dataset(dummy_data)
    print("\ncsi_cnn_dataset:")
    print("  Input shape:", test2[0][0].shape)   # (1, 2, 51)
    print("  Label shape:", test2[0][1].shape)   # ()
    print("  Dataset length:", len(test2))
    
    # Test quantized CSI dataset
    dummy_quan_data = np.random.randn(1000, 103)  # Quantized data
    test3 = csi_quan_dataset(dummy_quan_data)
    print("\ncsi_quan_dataset:")
    print("  Input shape:", test3[0][0].shape)   # (102,)
    print("  Label shape:", test3[0][1].shape)   # ()
    print("  Dataset length:", len(test3))