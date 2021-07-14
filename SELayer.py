import torch
import torch.nn as nn
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