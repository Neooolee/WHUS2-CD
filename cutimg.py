# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:03:22 2018

@author: Neoooli
"""

import numpy as np
import os
import glob
from gdaldiy import *

"""
The source file should be organized as follows:
./train/10m/file1.tif   (4 bands 2/3/4/8)
./train/20m/file1.tif   (6 bands 5/6/7/8A/11/12)
./train/60m/file1.tif   (3 bands 1/9/10)
"""
 
sourcedir='./train/'#image dir such as    
name='10m'#20m, 60m
sourcepath=sourcedir+name

def cut_data(filedir):
    window_size=384 #384 for 10 m bands, 192 for 20 m bands, 64 for 60 m bands.
    stride=384#384 for 10 m bands, 192 for 20 m bands, 64 for 60 m bands.
    filedirs=glob.glob(os.path.join(filedir, '*'))
    for i in range(len(filedirs)):
        filepath=filedirs[i]
        print(filepath)
        savedirname=filepath.split('\\')[-1].split('.tif')[0][27:44]
        savedirpath=filedir+'clips\\'+savedirname
        if not os.path.exists(savedirpath):
            os.makedirs(savedirpath)
        img = imgread(filepath)
        h,w=img.shape[0],img.shape[1]
        h_steps=(w-window_size)//stride+1
        w_steps=(h-window_size)//stride+1
        n=0
        for i in range(h_steps):
            high=i*stride
            for j in range(w_steps):
                width=j*stride
                n=n+1
                if np.all(img[high:high+window_size,width:width+window_size]>0):
                    imgwrite(savedirpath+"/"+str(n)+'.tif',img[high:high+window_size,width:width+window_size])
# cut_data(sourcepath)
def multi_dir(path):
    filedirs=glob.glob(os.path.join(path, '*'))
    for i in range(len(filedirs)):
        filedir=filedirs[i]       
        cut_data(filedir)
multi_dir(sourcepath)




