import torch
import torch.nn as nn
import torch.nn.functional as F
from pvt_v2 import pvt_v2_b2,pvt_v2_b2_li
import copy
import numpy as np
from collections import OrderedDict
from conmix_cbam import *
from T_cbam import *
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
        
class Conv2dD(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1):
        super(Conv2dD, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=1, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.conv = nn.Conv2d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        x=self.bn(x)
        x = self.relu(x)
        return x
        

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.network = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)

        )
        self.shortcut = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x+s)
        return x
class DSC(nn.Module):#Depthwise_Separable_Convolution
    def __init__(self, in_channel, out_channel, ksize=3,padding=1,bais=True):
        super(DSC, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,groups=in_channel,kernel_size=ksize,padding=padding,bias=bais)
        self.bn=nn.BatchNorm2d(in_channel)
        self.relu=nn.ReLU(inplace=True)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,padding=0,bias=bais)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.relu2=nn.ReLU(inplace=True)        
    def forward(self, x):
        out = self.depthwiseConv(x)
        out=self.bn(out)
        out=self.relu(out)
        out = self.pointwiseConv(out)
        out=self.bn2(out)
        out=self.relu2(out)        
        
        return out        
        
def kernel(num):
    kk= np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    w = np.expand_dims(kk, axis=0)
    w = np.expand_dims(w, axis=0)
    w = np.repeat(w, num, axis=0)
    tensorw = torch.from_numpy(w).float()
    return tensorw
def kernel2(num):
    kk= np.array([[0, -1, 0],
                   [-1,  4, -1],
                   [0, -1,0]])
    w = np.expand_dims(kk, axis=0)
    w = np.expand_dims(w, axis=0)
    w = np.repeat(w, num, axis=0)
    tensorw = torch.from_numpy(w).float()
    return tensorw
    
class Sharp_kernel(nn.Module):
    def __init__(self, in_chs):
        super(Sharp_kernel, self).__init__()
        self.weight=kernel(in_chs).cuda()
        self.in_chs=in_chs
    def forward(self, x):
        out=F.conv2d(x,self.weight,bias=None,stride=1,padding='same',groups=self.in_chs)
        return out
        
class Sharp_kernel2(nn.Module):
    def __init__(self, in_chs):
        super(Sharp_kernel2, self).__init__()
        self.weight=kernel2(in_chs).cuda()
        self.in_chs=in_chs
    def forward(self, x):
        out=F.conv2d(x,self.weight,bias=None,stride=1,padding='same',groups=self.in_chs)
        return out
class TSA(nn.Module):  # Spatial attention module
    def __init__(self,channels):
        super(TSA, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.conv2 = nn.Conv2d(1, 1, 7, 1, 3, bias=False)  # 7x7conv
        self.conv1 = nn.Conv2d(channels, 1,1,bias=False) 
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 
        x_conv=self.conv1(x)
        #x_avg=self.conv(x_avg)
        #x_max=self.conv2(x_max)
        
        xt=torch.cat([x_avg, x_max],dim=1)
        xs=self.conv(xt)
        #xs=x_max+x_avg
        
        return x*self.sigmoid(xs)                 
class Boundary_enhance(nn.Module):
    def __init__(self, in_chs):
        super(Boundary_enhance, self).__init__()
        self.sharp=Sharp_kernel(in_chs)
        self.sharp2=Sharp_kernel2(in_chs)
        self.conv=nn.Conv2d(in_chs, 1, kernel_size=1, bias=False)
        self.conv1=nn.Conv2d(in_chs,in_chs, kernel_size=1, bias=False)        
        self.sigmoid=nn.Sigmoid()
        self.dsc=DSC(in_chs,in_chs)
        self.relu=nn.ReLU(inplace=True)
        self.tsa=TSA(in_chs)
    def forward(self, x):
    
        out=self.sharp(x)
        #out2=self.sharp2(x)        
        xx=out+x#+out2
        #xx=self.relu(xx)
        out=self.dsc(xx)
        #out=self.tsa(out)
        return out#
        
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = residual_block(in_c[0]+in_c[1], out_c)
        self.r2 = residual_block(out_c, out_c)
        self.cc=ConvBNR(in_c[0]+in_c[1], out_c)
        self.c1=Conv1x1(in_c[0]+in_c[1], out_c)
        self.cx=ConvBNR(in_c[0], out_c)
        self.cbam=CBAM(channels=in_c[0]+in_c[1])
        self.BE=Boundary_enhance(out_c)
        self.tcbam=TCBAM(channels=in_c[0]+in_c[1])
    def forward(self, x, s):
        x = self.up(x)
        #x=x+s
        x = torch.cat([x, s], axis=1)
        x=self.cc(x)
        x=self.BE(x)        
        #x=self.cbam(x)
        return x
class decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = residual_block(in_c[0]+in_c[1], out_c)
        self.r2 = residual_block(out_c, out_c)
        self.cc=ConvBNR(in_c[0]+in_c[1], out_c)
        self.c1=Conv1x1(in_c[0]+in_c[1], out_c)
        self.cx=ConvBNR(in_c[0], out_c)
        self.cbam=CBAM(channels=in_c[0]+in_c[1])
        self.BE=Boundary_enhance(out_c)
        self.tcbam=TCBAM(channels=in_c[0]+in_c[1])
    def forward(self, x, s):
        x = self.up(x)
        #x=x+s
        x = torch.cat([x, s], axis=1)
        x=self.cc(x)
        #x=self.BE(x)        
        #x=self.cbam(x)
        return x
        
class senet(nn.Module):
    def __init__(self, in_channels):
        super(senet,self).__init__()
        
        #self.globalpool= F.adaptive_avg_pool2d(xx,(1,1))
        self.line1=torch.nn.Linear(in_channels,int(in_channels//16))
        self.relu=nn.ReLU(inplace=True)
        self.line2=torch.nn.Linear(int(in_channels//16),in_channels)
        self.sigmoid=nn.Sigmoid()
        #self.reshape=torch.reshape()       
    def forward(self, x):
        #print(x.shape)
        glb=F.adaptive_avg_pool2d(x,(1,1))
        glb=torch.squeeze(glb)
        #print(glb.shape)
        line1=self.line1(glb)
        relu=self.relu(line1)
        exc=self.line2(relu)
        #print(exc.shape)
        sigmoid=self.sigmoid(exc)
        exc=sigmoid.unsqueeze(-1)
        exc=exc.unsqueeze(-1)
        return exc#out
class DBA(nn.Module):
    def __init__(self, in_channels):
        super(DBA,self).__init__()
        
        self.GrouPconv=nn.Sequential(nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1,groups=16),nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))
        self.xcon2= nn.Conv2d(in_channels,in_channels,kernel_size=1,padding=0)
        self.xcon3 = nn.Conv2d(2,1,kernel_size=3,padding=1)
        self.sigmoid=nn.Sigmoid()
        self.dsc=DSC(in_channels,in_channels)
        self.in_channels=in_channels
        self.senet1=senet(in_channels)
        self.senet2=senet(in_channels)
        self.dsc=DSC(in_channels,in_channels)
        self.relu=nn.ReLU(inplace=True)
                               
    def forward(self, x):
        x=self.GrouPconv(x)   
        maxx,e=torch.max(x,dim=1)
        maxx=maxx.unsqueeze(1)
        softm1=self.sigmoid(maxx)

        out1=torch.mul(x,softm1)
        snt1=self.senet1(out1)
        out1=torch.mul(out1,snt1)        
                                
        meann=torch.mean(x,dim=1)
        meann=meann.unsqueeze(1)
        softm2=self.sigmoid(meann)

        out2=torch.mul(x,softm2)
        snt2=self.senet2(out2)
        out2=torch.mul(out2,snt2)                
        out=out1+out2
        #out=self.relu(out)        
        #out=self.xcon2(out)       
        return out
class BNet(nn.Module):
    def __init__(self,numclass):
        super(BNet,self).__init__()
                                     
        self.classifier1= nn.Sequential(nn.Linear(512**2,512))
        self.relu=nn.ReLU(inplace=True)
        self.classifiers = nn.Sequential(nn.Linear(512,numclass))        
        
    def forward(self,x,y):

        batch_size = x.size(0)
        feature_size = x.size(2)*x.size(3)
        x = x.view(batch_size , 512, feature_size)
        
        feature_size = y.size(2)*y.size(3)
        y= y.view(batch_size , 512, feature_size)
        
        x = (torch.bmm(x, torch.transpose(y, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x)*torch.sqrt(torch.abs(x)+1e-10))
        #print(x.shape)
        x = self.classifier1(x)
        x=self.relu(x)
        x = self.classifiers(x)
        return x
        
        
class pvtseg(nn.Module):
    def __init__(self, n_classes=1,num_classes=7):
        super(pvtseg, self).__init__()
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1),nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=1),nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        
        self.backbone = pvt_v2_b2_li()  # [64, 128, 320, 512]
        path = './pvt_v2_b2_li.pth'#pvt_v2_b2_li  pvt_v2_b2
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        
        #CNN
        self.inc = ConvBatchNorm(3, 16)
        self.Down1 = DownBlock(16, 16 * 2, nb_Conv=2)
        self.Down2 = DownBlock(16 * 2, 16 * 4, nb_Conv=2)

       
        self.c2 = nn.Conv2d(128, 64, 1, bias=False)#96
        self.c3 = nn.Conv2d(320, 64, 1, bias=False)#128
        self.c4 = nn.Conv2d(512, 64, 1, bias=False)#256
        
        self.be1=DBA(64)#Boundary_enhance(64)          
        self.be2=DBA(64)#Boundary_enhance(64)  
        self.be3=DBA(64)#Boundary_enhance(64)

        self.d1 = decoder_block([64, 64], 64)#[256, 128],128
        self.d2 = decoder_block([64, 64], 64)#[128, 96],96
        self.d3 = decoder_block([64, 64], 64)#[96, 64],64
        self.covdd=Conv2dD(128,64)
        
        self.out0 = nn.Conv2d(64, n_classes, 1)
        self.out1 = nn.Conv2d(64, n_classes, 1)
        self.out2 = nn.Conv2d(64, n_classes, 1)
        self.out3 = nn.Conv2d(64, n_classes, 1)#96
        self.out4 = nn.Conv2d(64, n_classes, 1)#128
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.cbam=CBAM(channels=512)
        self.tcbam=TCBAM(channels=512)
        self.BNet=BNet(num_classes)
        self.BE1=Boundary_enhance(64)
        self.tcbam1=TCBAM(channels=64)
        self.tcbam2=TCBAM(channels=64)
        self.tcbam3=TCBAM(channels=64)        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(nn.Linear(512, num_classes),)
        self.proj_head2 = nn.Sequential(nn.Linear(512, num_classes),) 
        
        self.conmxier=Conmixercm(dim=96,inclas=3,depth=6,n_classes=2)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # backbone
        # if grayscale input, convert to 3 channels
        x0=x
        if x.size()[1] == 1:
            x = self.conv(x)

        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        f2 = self.c2(x2)#96
        f3 = self.c3(x3)#128
        f4 = self.c4(x4)#256

        f1=x1#64
        
        d1 = self.d1(f4, f3)
        #d1=self.be1(d1)
        d2 = self.d2(d1, f2)
        #d2=self.be2(d2)
        d3 = self.d3(d2, f1)
        #d3=self.be3(d3)

        pre3= self.out2(d3)
        pre2= self.out3(d2)
        pre1= self.out4(d1) 
                             
        p3 = F.interpolate(pre3, scale_factor=4, mode='bilinear')
        p2 = F.interpolate(pre2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(pre1, scale_factor=16, mode='bilinear')
        #p0 = F.interpolate(pre0, scale_factor=32, mode='bilinear')
        
        pp=p3+p2+p1
        
        xm = torch.cat([pp, x0], axis=1)####classify        
        xm = self.conv2(xm)
        
        
        xm=self.conmxier(xm)
        
        #xn=self.BNet(pvt[3],pvt[3])
        
        ppt=self.tcbam(pvt[3])
        xn = self.avgpool(ppt)###fusion
        xn= torch.flatten(xn, 1)
        xn = self.proj_head2(xn)  
              
        return pp,xn,xm
        
#x = torch.randn((4, 3,224, 224))
#mm=pvtseg(num_classes=4)(x)
# print(mm[4].shape)





