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
testlist=["S2A_MSIL1C_20180930T030541_N0206_R075_T49QDD_20180930T060706",
"S2A_MSIL1C_20191105T023901_N0208_R089_T51STR_20191105T054744",
"S2A_MSIL1C_20190812T032541_N0208_R018_T48RXU_20190812T070322",
"S2A_MSIL1C_20190602T021611_N0207_R003_T52TES_20190602T042019",
"S2A_MSIL1C_20190328T033701_N0207_R061_T49TCF_20190328T071457",
"S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939",
"S2A_MSIL1C_20200416T042701_N0209_R133_T46SFE_20200416T074050",
"S2A_MSIL1C_20200528T050701_N0209_R019_T44SPC_20200528T082127",
"S2A_MSIL1C_20210207T023851_N0209_R089_T52UCU_20210207T040210",
"S2A_MSIL1C_20210126T052111_N0209_R062_T44SNE_20210126T063836",
"S2A_MSIL1C_20210102T054231_N0209_R005_T43SFB_20210102T065941",
"S2A_MSIL1C_20201206T041141_N0209_R047_T47SMV_20201206T053320"]

sourcedir='F:/WHU/WHUS2-CD+/'#sourcedir
name='unziped'
sourcepath=sourcedir+name#the unziped file path
savepath=sourcedir#savepath
bands = [['10m','02','03','04','08'],['20m','05','06','07','8A','11','12'],['60m','01','09','10']]
def fuse_DN(path1):
    filename=path1.split('\\')[-1].split('.SAFE')[0]#[33:44]
    filename = os.path.basename(filename)
    if filename in testlist:
        savepath=sourcedir+"test"
    else:
        savepath=sourcedir+"train"
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
    filedirs=glob.glob(os.path.join(path, 'S2*'))
    for i in range(len(filedirs)):
        filedir=filedirs[i]
        print(filedir)    
        fuse_DN(filedir)
multi_dir(sourcepath)
os.system("python cutimg.py")
