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
The source file should be organized as follows by running fuseimg.py first:
sourcedir
        /10m/file1.tif...filen.tif   (4 bands 2/3/4/8)
        /20m/file1.tif...filen.tif   (6 bands 5/6/7/8A/11/12)
        /60m/file1.tif...filen.tif   (3 bands 1/9/10)
"""
 
sourcedir='F:/WHU/WHUS2-CD+/composite/'#source dir  
names=['10m','20m','60m']
window_sizes,strides=[384,192,64],[384,192,64]

def cut_data(filedir,window_size,stride):
    filedirs=glob.glob(os.path.join(filedir, '*'))
    for i in range(len(filedirs)):
        filepath=filedirs[i]
        print(filepath)
        savedirname=filepath.split('\\')[-1].split('.tif')[0][33:44]
        savedirpath=filedir.replace("composite",'clips')+"/"+savedirname
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
def multi_dir(filedir,window_size,stride):
    print(filedir)      
    cut_data(filedir,window_size,stride)
for i in range(len(names)):  
    multi_dir(sourcedir+names[i],window_sizes[i],strides[i])




