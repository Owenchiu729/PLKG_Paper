import torch.nn as nn
import torch
# import PyTorch chneural network model 

class cnn_basic(nn.Module):
    def __init__(self):
        super(cnn_basic,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,4,kernel_size=(2,3),padding=(0,1)),
                                   nn.LayerNorm([4,1,51]),
                                   nn.ReLU())
        #
        self.conv2 = nn.Sequential(nn.Conv2d(4,1,kernel_size=(1,1)),
                                   nn.LayerNorm([1,1,51]),
                                   nn.ReLU())
        self.fnn = nn.Sequential(nn.Linear(51,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,32),
                                 nn.LayerNorm(32),
                                 nn.ReLU(),
                                 nn.Linear(32,51),
                                 nn.LayerNorm(51),   
                                 nn.Sigmoid()) 
# Define CNN+FNN model for CSI feature extraction and processing
# 定義 CNN+FNN 模型，用於 CSI 特徵抽取與處理

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out2 = torch.squeeze(out2)
        out = self.fnn(out2)
# Forward pass: 
# 1. conv1 extracts local CSI features
# 2. conv2 fuses channels
# 3. squeeze flattens to vector
# 4. fnn maps to final 51-dim features
        return out

if __name__ == "__main__":
    x = torch.rand(1,2,51)
    test = cnn_basic()
    hold = test(x)
    print(hold.size())

# 這段程式碼是主程式測試區 (main block)，
# 只有當這個檔案被「直接執行」時才會執行，不會在被 import 時自動跑。
#
# Step 1. 建立隨機輸入資料 x = torch.rand(1,1,2,51)
#   - 模擬 CSI 特徵數據
#   - tensor 格式為 (batch=1, channel=1, height=2, width=51)
#   - 這符合 Conv2d 所需的輸入維度 (N, C, H, W)
#
# Step 2. 建立模型 test = cnn_basic()
#   - 實例化你定義的 CNN+FNN 模型
#
# Step 3. hold = test(x)
#   - 把隨機 CSI 輸入丟進模型
#   - 這會自動呼叫 forward()，執行 conv1 → conv2 → squeeze → fnn
#
# Step 4. print(hold.size())
#   - 印出模型最終輸出的 shape
#   - 預期輸出大小為 torch.Size([1, 51])，代表 51 維特徵向量
#