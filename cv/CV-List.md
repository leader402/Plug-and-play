[toc]
### 1. Polarized Self-Attention: Towards High-quality Pixel-wise Regression

**Main idea:通道注意力+空间注意力**

![image](https://cdn.jsdelivr.net/gh/leader402/image@main/image/screenShots/1626273404691-1626273404683-_20210714221842.jpg)

Paper:https://arxiv.org/pdf/2107.00782.pdf

Code1:https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/attention/PolarizedSelfAttention.py

Code2:https://github.com/DeLightCMU/PSA   

### 2.Squeeze-and-Excitation Network(SELayer)

**main idea：通道注意力机制，对通道（channel）进行加权**

![image-20210714230148274](./pic/image-20210714230148274.png)

![image-20210714230249198](./pic/image-20210714230249198.png)

Paper:https://arxiv.org/pdf/1709.01507.pdf

Code:https://github.com/hujie-frank/SENet
### 4.Self-attention CV
![image](https://user-images.githubusercontent.com/34624932/126118169-b1153135-b2f4-477f-a2ad-4cc4802cf649.png)
from：https://zhuanlan.zhihu.com/p/283125663

### 12. CBAM: Convolutional Block Attention Module

**Main idea:通道注意力(双池化+MLP)+空间注意力**

![CBAM](./pic/CBAM.png)

![CBAM](./pic/CBAM2.png)

Paper:https://arxiv.org/abs/1807.06521

Code1:https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

