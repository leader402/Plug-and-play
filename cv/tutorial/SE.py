import torch
import torch.nn as nn
#--------------SELayer define and test---------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1, x2, _, __ = x.size()
        y = self.avg_pool(x).view(x1, x2)
        y = self.fc(y).view(x1, x2, 1, 1)
        y_out = y.expand_as(x)#匹配x
        return x * y_out
#测试
if __name__ == '__main__':
    a=torch.randn(1,64,128,128)
    selayer=SELayer(64)
    print(selayer(a).shape)
#--------------tutorial---------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,(5, 5))
        self.conv2 = nn.Conv2d(6,16,(5, 5))
        #conv2：input_channel-6，output_channel-16
        #SELayer：input_channel-16，output_channel-16 
        #-----------SELayer_define-----------
        self.SELayer=SELayer(16)
        #-----------SELayer_define-----------
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2, 2))
        #-----------SELayer_forward-----------
        x = self.SELayer(x)         
        #-----------SELayer_forward-----------
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
#--------------tutorial---------------
        