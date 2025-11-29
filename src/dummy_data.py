import os
import datetime
import numpy as np

#=======================================
# Config 
#=======================================
N=1000  # number of CSI samples
subcarriers=52  # 0 = RSSI, 1~51 = CSI subcarriers
save_dir="../data"
os.makedirs(save_dir,exist_ok=True)# save under project_root/data

#=======================================
# Generate dummy CSI data
#=======================================
# Create a 2D array with shape (N, 52) for a single device: ESP32
data = np.zeros((N, subcarriers), dtype=np.float32) 

# Column 0 -> Simulated RSSI values (mean=-55, std=5 dB)
data[:, 0] = -55 + 5 * np.random.randn(N) 

# Columns 1~51 -> CSI amplitude values
# Using cosine wave + small random noise to simulate realistic CSI patterns
data[:, 1:] = np.cos(np.linspace(0, 10, N * 51)).reshape(N, 51) + np.random.randn(N, 51) * 0.1



#=======================================
# Save file with timestamp
#=======================================
timestamp =datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename=f"esp32_csi_dataset_{timestamp}.npy"
save_path =os.path.join(save_dir, filename)

#Save as .npy file
np.save(save_path, data)
print('Dummy CSI dataset generated for ESP32')
print(f'Shape = {data.shape}')