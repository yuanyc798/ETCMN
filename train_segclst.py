import os
import sys
from tqdm import tqdm
#from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
from numpy import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torchvision.utils import make_grid
from torch.nn.modules.loss import CrossEntropyLoss,BCEWithLogitsLoss,BCELoss
from imgaug import augmenters as iaa
import math
from dataset import Lesion_Dataset
from torch.nn import  MSELoss
#import ramps, losses
#from process import RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import torchvision
from sdf import *
#from PFD import *
from pvtseg import *
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./breastseg/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=20, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=10, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.2, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=50, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path

from glob import glob
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0')

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
def dice_loss(score, target):
    score=torch.sigmoid(score)
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target,dim=(2,3))
    y_sum = torch.sum(target * target,dim=(2,3))
    z_sum = torch.sum(score * score,dim=(2,3))
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - torch.mean(loss)
    return loss
def dice_m(score, target, smooth=1e-10):
    intersect = torch.sum(score * target,dim=(2,3))
    y_sum = torch.sum(target * target,dim=(2,3))
    #print(torch.sum(y_sum).item())
    z_sum = torch.sum(score * score,dim=(2,3))
    dc= (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return torch.mean(dc.float())
    
def preprocess_input(x):#BGR
    #x = skimage.color.rgb2gray(x) 
    x = (x - np.mean(x)) / np.std(x)
    return x
    
def prepro_input(x):
    mn=torch.mean(x,dim=[1,2,3])
    std=torch.std(x,dim=[1,2,3])
    mn=torch.unsqueeze(mn,1)
    mn=torch.unsqueeze(mn,2)
    mn=torch.unsqueeze(mn,3)
    
    std=torch.unsqueeze(std,1)
    std=torch.unsqueeze(std,2)
    std=torch.unsqueeze(std,3)
    
    #print(mn.shape,std.shape,x.shape)
    x=(x-mn)/std
    return x
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
        
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def validate(net, valid_loader):
        # validation steps
        with torch.no_grad():
            net.eval()
            valid_dc = 0
            loss = 0
            num=0
            for images, labels,numclass in valid_loader:
                images = images.cuda().float()
                labels = labels.cuda().float()
                numclass =numclass.cuda().float()
                #images=prepro_input(images)
                #labels = labels.to(device,dtype=torch.float32)
                pred,ccls,sdff= net(images)
                loss += structure_loss(pred, labels)+loss_func(ccls,numclass.long())
                valid_dc += dice_m(torch.sigmoid(pred),labels)
                
                xx=np.argmax(F.softmax(ccls, dim=1).data.cpu().numpy(), axis = 1)
                nnm=np.array(xx,dtype=np.float32)-np.array(numclass.cpu().numpy(),dtype=np.float32)
                nm=len(nnm[nnm==0])
                num+=nm
                
        return valid_dc/len(valid_loader),num/116,loss/len(valid_loader)
def lab_path(file_path):
      jj=file_path.split('/')[-1].split('.')[0][-1]
      #print(jj)
      if jj=='B':
        lm=0
      elif jj=='M':
        lm=1

      return lm
def test_net(net,device,tst_path):
        state_dict = torch.load('./best_model.pth')
        net.load_state_dict(state_dict)
        #tst_path='/storage/yyc/data/tst/'
        listname=glob(tst_path+'*.jpg')
        path_save=r'./save/segcls/'
        isExists=os.path.exists(path_save)
        if not isExists:
           os.makedirs(path_save)
        llen=0
        
        pas='./save/'
        f=open(pas+'segcls.txt','w')
        with torch.no_grad():
            net.eval()
            for image_path in listname:
                image = cv2.imread(image_path)
                #image =cv2.resize(image,(224,224),interpolation=cv2.INTER_LINEAR)
                image = np.array(image)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image=preprocess_input(image)
                image=image.reshape(1,image.shape[0],image.shape[1])
                image=np.expand_dims(image, axis=0)
                image=torch.from_numpy(image)
                images= image.cuda().float()

                 
                y1,y2,y3= net(images)
                
                y1=torch.squeeze(torch.sigmoid(y1))
                #y = F.softmax(y1, dim=1)
                mm = y1.cpu().data.numpy()
                mm[mm>=0.5]=1#mmg=y[0,:,:,:]
                mm[mm<0.5]=0#mg=np.squeeze(np.argmax(mmg,0))
                cv2.imwrite(path_save+image_path.split('/')[-1].split('.')[0]+'.png',mm*255)
                 
                y= F.softmax(y2, dim=1)
                y = y.cpu().data.numpy()
                abb=np.float32(np.argmax(y[0]))-np.float32(lab_path(image_path))
                if abb==0:
                  llen+=1
                  
                sv=str(np.argmax(y[0]))
                f.write(image_path.split('/')[-1])
                f.write(',')
                f.write(sv)
                f.write('\n')
        f.close()
                  
        return llen/213

loss_func =nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')
   
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()
if __name__ == "__main__":

    def create_model(ema=False):
        # Network definition
        #net =SwinTransformer(img_size=224,window_size=7,num_classes=7).to(device)
        #net=vgg19.to(device)
        #net=BNet(7).to(device)
        #net=SwinTransformerSys(img_size=(224,224),window_size=7,num_classes=7)
        #net=PFD(num_classes=2)
        net=pvtseg(num_classes=2)
        #net=ConvMixer(dim=96,depth=6,n_classes=7).to(device)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    db_train= Lesion_Dataset(train_data_path+'train/',True,True)                   
    db_val= Lesion_Dataset(train_data_path+'val/',True,False)

    batch_size=16
    trainloader = DataLoader(db_train, batch_size,shuffle = True)
    val_loader=DataLoader(db_val,batch_size,shuffle = False)
    
    model.train()
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer =optim.Adam(model.parameters(),lr=base_lr, betas=(0.9, 0.999), eps=1e-08)
    ce_loss =F.cross_entropy#F.cross_entropy#CrossEntropyLoss()#()#BCEWithLogitsLoss()#
    #dice_loss =losses.dice_loss#losses.DiceLoss(2)#DiceLoss(2)
    mse_loss = MSELoss()

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    val_dice=0
    val_ls=2
    epochs=100
    #pathsave=''
    consistency_rampup=len(trainloader)
    for epoch_num in tqdm(range(epochs), ncols=100):
        time1 = time.time()
        lossm=0
        for  i,(image,label,numclass) in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch = image,label#sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch =volume_batch.cuda().float(),label_batch.cuda().float()
            numclass=numclass.cuda().float()
            
            outputs,conts,outc= model(volume_batch)#.requires_grad_()
            loss_dice = structure_loss(outputs, label_batch)

            c_loss = loss_func(conts,numclass.long())
            n_loss = loss_func(outc,numclass.long())
            
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:, 0, ...].cpu().numpy(), outputs[:, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()

            consis_loss = torch.mean((conts-outc)**2)            
            #consis_loss = kl_loss(torch.log_softmax(conts, dim=1), torch.softmax(outc, dim=1))            
                                              
            
            consistency_weight = get_current_consistency_weight(iter_num/epochs)
            loss = loss_dice+0.3*(c_loss+n_loss)+ consistency_weight * consis_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossm+=loss.item()
            #update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
                # param_group['lr'] = lr_

            iter_num = iter_num + 1
        v_dc,v_acc,val_loss=validate(model,val_loader)
        print('epoch:%d  train loss:%f' % (epochs+1,lossm/len(trainloader)))
        if v_dc>val_dice:
                print('val_dc changed:',v_dc.item(),'val_acc:',v_acc,'model saved, val_loss:',val_loss.item())
                
                val_dice=v_dc
                torch.save(model.state_dict(), './best_model.pth')
    tst_path = r'./tst/'
    ac=test_net(model,device,tst_path)
    print('tst ac',ac)
    #save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    #torch.save(model.state_dict(), save_mode_path)
    #logging.info("save model to {}".format(save_mode_path))
    #writer.close()
