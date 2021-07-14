# 1.Two_Conv代表两个卷积层拼接
# 2.downsample代表下采样层
# 3.upsample代表上采样层
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
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = Two_Conv(n_channels, 64)
        self.downsample1 = downsample(64, 128)
        self.selayer1=SELayer(128)#插入方式，确保前后维度一致
        self.downsample2 = downsample(128, 256)
        self.downsample3 = downsample(256, 512)
        factor = 2 if bilinear else 1
        self.downsample4 = downsample(512, 1024 // factor)
        self.upsample1 = upsample(1024, 512 // factor, bilinear)
        self.upsample2 = upsample(512, 256 // factor, bilinear)
        self.upsample3 = upsample(256, 128 // factor, bilinear)
        self.upsample4 = upsample(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.downsample1(x1)
        x2 = self.selayer1(x2)#插入方式
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)
        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        logits = self.outc(x)
        return logits