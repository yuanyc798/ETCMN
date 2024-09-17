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
        
class convq(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(convq, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,stride=5,kernel_size=3,padding=0)
        self.softmx=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.conv(x)
        return x
class ChannelAttention(nn.Module):  # Channel attention module
    def __init__(self, channels, ratio=16):  # r: reduction ratio=16
        super(ChannelAttention, self).__init__()

        hidden_channels = channels // ratio
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # global avg pool
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # global max pool
        self.convpool =convq(channels,channels)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False),)
        self.mlp2 = nn.Sequential(nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=False),)            
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        #x_conv =self.convpool(x)
        avg=self.mlp(x_avg)
        max=self.mlp2(x_max)
        #conv=self.mlp2(x_conv)
        add=avg + max#+conv
        #print(conv.shape)
        #print(avg.shape)
        return self.sigmoid(add)  # Mc(F) = σ(MLP(AvgPool(F))+MLP(MaxPool(F)))= σ(W1(W0(Fcavg))+W1(W0(Fcmax)))�?


class SpatialAttention(nn.Module):  # Spatial attention module
    def __init__(self,channels):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(1, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.conv2 = nn.Conv2d(1, 1, 7, 1, 3, bias=False)        
        #self.conv1 = nn.Conv2d(channels, 1,1,bias=False) 
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 
        x_avg=self.conv(x_avg)
        x_max=self.conv2(x_max)
        xs=x_avg+x_max        
        
        #x_conv=self.conv1(x)
        
        return self.sigmoid(xs)       
        #return self.sigmoid(self.conv(torch.cat([x_avg, x_max,x_conv],dim=1)))  


class TCBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, channels, ratio=16):
        super(TCBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, ratio)  # Channel attention module
        self.spatial_attention = SpatialAttention(channels)  # Spatial attention module

    def forward(self, x):
        f1 = self.channel_attention(x) * x  # 
        f2 = self.spatial_attention(f1) * f1  # 
        return f2
