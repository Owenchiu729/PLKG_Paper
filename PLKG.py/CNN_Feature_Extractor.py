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
# �w�q CNN+FNN �ҫ��A�Ω� CSI �S�x����P�B�z

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

# �o�q�{���X�O�D�{�����հ� (main block)�A
# �u����o���ɮ׳Q�u��������v�ɤ~�|����A���|�b�Q import �ɦ۰ʶ]�C
#
# Step 1. �إ��H����J��� x = torch.rand(1,1,2,51)
#   - ���� CSI �S�x�ƾ�
#   - tensor �榡�� (batch=1, channel=1, height=2, width=51)
#   - �o�ŦX Conv2d �һݪ���J���� (N, C, H, W)
#
# Step 2. �إ߼ҫ� test = cnn_basic()
#   - ��ҤƧA�w�q�� CNN+FNN �ҫ�
#
# Step 3. hold = test(x)
#   - ���H�� CSI ��J��i�ҫ�
#   - �o�|�۰ʩI�s forward()�A���� conv1 �� conv2 �� squeeze �� fnn
#
# Step 4. print(hold.size())
#   - �L�X�ҫ��̲׿�X�� shape
#   - �w����X�j�p�� torch.Size([1, 51])�A�N�� 51 ���S�x�V�q
#