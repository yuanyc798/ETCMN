# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 :28
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
def preprocess_input(x):#BGR
    #x = skimage.color.rgb2gray(x) 
    x = (x - np.mean(x)) / np.std(x)
    return x
def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential([iaa.Fliplr(0.4),iaa.Flipud(0.3),sometimes(iaa.Crop()),#sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
                          sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},rotate=(-45, 45),shear=(-15, 15),cval=0))])
    return seq 
def lab_path(file_path):
      jj=file_path.split('/')[-1].split('.')[0][-1]
      #print(jj)
      if jj=='B':
        lm=0
      elif jj=='M':
        lm=1
        
      return lm
class Lesion_Dataset(Dataset):
    def __init__(self,data_path,TRUE,aug):
        self.data_path = data_path
        self.aug=aug
        if TRUE is True:
            self.image_path = glob.glob(os.path.join(data_path,'*.jpg'))
        else:
            self.image_path = glob.glob(os.path.join(data_path,'*.jpg'))[0:400]

    def augment(self,image,mode):
        """
        :param image:
        :param mode: 1 :
        """
        file = cv2.flip(image,mode)
        return file
    def __len__(self):
        #print('ok')
        return len(self.image_path)

    def __getitem__(self,index):
        image_path = self.image_path[index]
        label_path = image_path.replace("jpg","png")

        #name=image_path.split('\\')[1]
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        labelclass =lab_path(image_path)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=preprocess_input(image)        
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        label = label / 255
        seq = get_seq()
        if self.aug:
                seq_det = seq.to_deterministic()
                X_aug =seq_det.augment_image(image)               
                y =ia.SegmentationMapsOnImage(np.squeeze(label).astype(np.uint8), shape=(320,320,1))
                Y_aug =seq_det.augment_segmentation_maps([y])[0].get_arr().astype(np.uint8) 
                image= X_aug
                label= Y_aug
        
        image = image.reshape(1,image.shape[0],image.shape[1])
        label = label.reshape(1,label.shape[0],label.shape[1])
        image=image.copy()
        label=label.copy()
        #255 -> 1


        #mode = random.choice([-1,0,1,2])
        #if mode != 2:
            #image = self.augment(image,mode)
            #label=self.augment(label,mode)
        return image, label,labelclass#,name

