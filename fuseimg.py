# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:03:22 2018

@author: Neoooli
"""

import numpy as np
import os
import glob
from gdaldiy import *

'''
The source zip file should be unziped and organized as follows:
sourcedir
        /S2A_MSIL1C_20180429T032541_N0206_R018_T49SCV_20180429T062304.SAFE
                                                                          /AUX_DATA...
        /S2A_MSIL1C_20180722T030541_N0206_R075_T49RFP_20180722T060550.SAFE
                                                                          /AUX_DATA...
'''
sourcedir='F:/WHU/WHUS2-CD+/'#sourcedir
name='unziped'
sourcepath=sourcedir+name#the unziped file path
savepath=sourcedir+"composite"#savepath
bands = [['10m','02','03','04','08'],['20m','05','06','07','8A','11','12'],['60m','01','09','10']]
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

def multi_dir(path):
    filedirs=glob.glob(os.path.join(path, '*'))
    for i in range(len(filedirs)):
        filedir=filedirs[i]
        print(filedir)    
        fuse_DN(filedir)
multi_dir(sourcepath)
os.system("python cutimg.py")
