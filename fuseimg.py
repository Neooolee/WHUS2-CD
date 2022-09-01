# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:03:22 2018

@author: Neoooli
"""

import numpy as np
import os
import glob
from gdaldiy import *

sourcedir='./'#原图文件夹路径
name='trainunziped'
sourcepath=sourcedir+name
savedir='./'#原图文件夹路径
savename='6bandsDN'
savepath=savedir+savename
bandNGB=['08','03','02']
bandRGB=[['10m','08','03','02']]
bands = [['10m','02','03','04','08'],['20m','05','06','07','8A','11','12'],['60m','01','09','10']]
rbands=[['6bands','02','03','04','08','11','12']]
def liner_2(input_):#2%线性拉伸,返回0~1之间的值
    def strech(img):
        low,high=np.percentile(img,(2,98))
        img[low>img]=low
        img[img>high]=high
        return (img-low)/(high-low)
    if len(input_.shape)>2:
        for i in range(input_.shape[-1]):
            input_[:,:,i]=strech(input_[:,:,i])
    else:
        input_=strech(input_)    
    return input_
def fuse_rgbstrech(path1):
    filename=path1.split('\\')[-1]
    DN_img = imgread(path1)[:,:,[3,1,2]]
    nodataindex=[DN_img==0]
    DN_img[DN_img>10000]=10000
    rescaled_img=DN_img/10000
    rescaled_img=liner_2(rescaled_img)
    rescaled_img[nodataindex]=0  
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imgwrite(savepath+"/"+filename+'.tif',np.uint8((rescaled_img)*255))
# filedir1=glob.glob(os.path.join('./rr', '*'))
# for i in range(len(filedir1)):
#     fuse_rgbstrech(filedir1[i])
def fuse_strech(path1):
    filedir1=glob.glob(os.path.join(path1, 'GRANULE'))[0]
    filedir2=glob.glob(os.path.join(filedir1, 'L*'))[0]
    filedir3=glob.glob(os.path.join(filedir2, 'IMG_DATA'))[0]
    # filedir3=glob.glob(os.path.join(filedir3, 'R10m'))[0]
    filename=path1.split('\\')[-1].split('.SAFE')[0]#[33:44]
    print(filename)
    Img_concat=[]
    for i in range(len(bands)):
        file=glob.glob(os.path.join(filedir3, '*'+str(bandNGB[i])+'.jp2'))[0]
        img = imgread(file)
        DN_img=img[:,:,np.newaxis] 
        nodataindex=[DN_img==0]
        DN_img[DN_img>10000]=10000
        rescaled_img=DN_img/10000
        rescaled_img=liner_2(rescaled_img)
        rescaled_img[nodataindex]=0
        Img_concat.append(rescaled_img)
    fusedimg = np.concatenate(Img_concat,axis=2)    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imgwrite(savepath+"/"+filename+'.tif',np.uint8((fusedimg)*255))
def fuse_strechL2A(path1):
    filename=path1.split('\\')[-1].split('.SAFE')[0][33:44]
    print(filename)
    filedir1=glob.glob(os.path.join(path1, 'GRANULE'))[0]
    filedir2=glob.glob(os.path.join(filedir1, 'L*'))[0]
    filedir3=glob.glob(os.path.join(filedir2, 'IMG_DATA'))[0]
    for k in range(len(bandRGB)):
        if os.path.exists(savedir+"/"+filename+'.tif'):
            continue
        filedir4=glob.glob(os.path.join(filedir3, 'R'+str(bandRGB[k][0])))[0]
        Img_concat=[]
        for i in range(1,len(bandRGB[k])):
            filepath=glob.glob(os.path.join(filedir4, '*'+str(bandRGB[k][i])+'_'+str(bandRGB[k][0])+'.jp2'))[0]
            img = imgread(filepath)
            DN_img=img[:,:,np.newaxis]
            nodataindex=[DN_img==0]
            DN_img[DN_img>10000]=10000
            rescaled_img=DN_img/10000
            rescaled_img=liner_2(rescaled_img)
            rescaled_img[nodataindex]=0
            Img_concat.append(rescaled_img)
    fusedimg = np.concatenate(Img_concat,axis=2)    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imgwrite(savepath+"/"+filename+'.tif',np.uint8((fusedimg)*255))
def fuse_DN(path1):
    filename=path1.split('\\')[-1].split('.SAFE')[0]#[33:44]
    print(filename)
    filedir1=glob.glob(os.path.join(path1, 'GRANULE'))[0]
    filedir2=glob.glob(os.path.join(filedir1, 'L*'))[0]
    filedir3=glob.glob(os.path.join(filedir2, 'IMG_DATA'))[0]
    for k in range(len(bands)):
        savedir=savepath+"/"+str(bands[k][0])    
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if os.path.exists(savedir+"/"+filename+'.tif'):
            continue
        Img_concat=[]
        for i in range(1,len(bands[k])):
            filepath=glob.glob(os.path.join(filedir3, '*'+str(bands[k][i])+'.jp2'))[0]
            img = imgread(filepath)
            DN_img=img[:,:,np.newaxis]
            Img_concat.append(DN_img)
        fusedimg = np.concatenate(Img_concat,axis=2)
        print(fusedimg.shape)

        imgwrite(savedir+"/"+filename+'.tif',fusedimg)
import cv2
def rfuse_DN(path1):
    filename=path1.split('\\')[-1].split('.SAFE')[0]#[33:44]
    print(filename)
    filedir1=glob.glob(os.path.join(path1, 'GRANULE'))[0]
    filedir2=glob.glob(os.path.join(filedir1, 'L*'))[0]
    filedir3=glob.glob(os.path.join(filedir2, 'IMG_DATA'))[0]
    for k in range(len(rbands)):
        savedir=savepath+"/"+str(rbands[k][0])    
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if os.path.exists(savedir+"/"+filename+'.tif'):
            continue
        Img_concat=[]
        for i in range(1,len(rbands[k])):
            filepath=glob.glob(os.path.join(filedir3, '*'+str(rbands[k][i])+'.jp2'))[0]
            img = imgread(filepath)
            if rbands[k][i]=='11' or rbands[k][i]=='12':
                img=cv2.pyrUp(img)
            DN_img=img[:,:,np.newaxis]
            Img_concat.append(DN_img)
        fusedimg = np.concatenate(Img_concat,axis=2)
        print(fusedimg.shape)

        imgwrite(savedir+"/"+filename+'.tif',fusedimg)
def multi_dir(path):
    filedirs=glob.glob(os.path.join(path, '*'))
    for i in range(len(filedirs)):
        filedir=filedirs[i]
        print(filedir)    
        # fuse_strech(filedir)
        rfuse_DN(filedir)
multi_dir(sourcepath)
# s=imgread("./6.tif")
# imgwrite("./allcloud.tif",np.ones_like(s)*4096)
# fuse_data('E:\\lijun\\data\\graduatedata\\clouddetection\\S2A\\test\\testunziped\\S2A_MSIL1C_20180827T032541_N0206_R018_T48RYV_20180827T062627.SAFE\\')
