#https://blog.csdn.net/Peach_____/article/details/128723630
import torch
from torch import nn
import torch.nn.functional as F
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class ChannelAttention(nn.Module):  # Channel attention module
    def __init__(self, channels, ratio=16):  # r: reduction ratio=16
        super(ChannelAttention, self).__init__()

        hidden_channels = channels // ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # global avg pool
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # global max pool
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False),)
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        avg=self.mlp(x_avg)
        max=self.mlp(x_max)
        add=avg + max
        return self.sigmoid(add)  # Mc(F) = σ(MLP(AvgPool(F))+MLP(MaxPool(F)))= σ(W1(W0(Fcavg))+W1(W0(Fcmax)))�?


class SpatialAttention(nn.Module):  # Spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 
        return self.sigmoid(
            self.conv(torch.cat([x_avg, x_max],dim=1))
        )  # Ms(F) = σ(f7×7([AvgP ool(F);MaxPool(F)])) = σ(f7×7([Fsavg;Fsmax]))�?
print(2**3)

class CBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, channels, ratio=16):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention = SpatialAttention()  # Spatial attention module

    def forward(self, x):
        f1 = self.channel_attention(x) * x  # 
        f2 = self.spatial_attention(f1) * f1  # 
        return f2
class DCBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, channels, ratio=16):
        super(DCBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention = SpatialAttention()  # Spatial attention module
        
        self.channel_attention2 = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention2 = SpatialAttention()         

    def forward(self, x):
        f1 = self.channel_attention(x) * x  # 
        f2 = self.spatial_attention(x) * x  #
        xc=f1+f2

        d1 = self.spatial_attention2(xc) *xc  #         
        d2 = self.channel_attention2(d1) * d1  # 
                
        return d2        
        
class Conmixercm(nn.Module):  # MLP Head
    def __init__(self,dim, depth, inclas,kernel_size=7, patch_size=7, n_classes=7):
        super(Conmixercm, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(inclas, dim, kernel_size=patch_size, stride=patch_size),nn.GELU(),
                  nn.BatchNorm2d(dim),)
        self.ss=nn.Sequential(Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=3),nn.GELU(),
                    nn.BatchNorm2d(dim))),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim))
        self.ada=nn.AdaptiveAvgPool2d((1,1))
        self.fla=nn.Flatten()
        self.lin=nn.Linear(64, n_classes)
        self.depth=depth

        self.lin1=nn.Linear(dim, n_classes)
        self.cbam=CBAM(channels=dim)
        self.dim=dim
        self.mp=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        x =self.conv(x)
        #print(x.shape)
        for i in range(self.depth):
          x=self.ss(x)
        #x=self.mp(x)
        #print(x.shape)
        #x=self.cbam(x)
        
        x=self.ada(x)
        x=self.fla(x)
        #xc=torch.cat([x, res34], dim=1) 
        x=self.lin1(x)
        #x=self.lin(x)
        return x
x = torch.randn((4, 3, 224, 224))
uu=Conmixercm(dim=96,inclas=3,depth=6,n_classes=2)(x)