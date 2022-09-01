# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:03:22 2018

@author: Neoooli
"""

import numpy as np
import os
import glob
from gdaldiy import *


import subprocess
import zipfile
import os
# from fuseimg import *

def unzip_file(zip_file_name,zip_output_dir, mode='rb'):

    zip_file = open(zip_file_name, mode)

    zip_fn = zipfile.ZipFile(zip_file)

    namelist = zip_fn.namelist()
    for item in namelist:

        zip_fn.extract(item, zip_output_dir)

    zip_fn.close()
    zip_file.close()    

    print("Unzipping finished!")

    return namelist[0]
       
origin_dir = "F:/WHU/WHUS2-CD+/aa/"
pattern = ".zip"
name='unziped'
zippedoutput_dir =origin_dir+name
if not os.path.exists(zippedoutput_dir): #如果路径不存在即创建
      os.makedirs(zippedoutput_dir)

for in_file in os.listdir(origin_dir):

    if pattern in in_file:

        zip_file_path = os.path.join(origin_dir, in_file)

        safe_in_file_path = unzip_file(zip_file_path,zippedoutput_dir)
        print("{} processing finished!\n".format(safe_in_file_path))

print("All zipped file finished!") 
'''
After run unzip.py, the dataset will be organized as follows:
sourcedir
        /S2A_MSIL1C_20180429T032541_N0206_R018_T49SCV_20180429T062304.SAFE
                                                                          /AUX_DATA...
        /S2A_MSIL1C_20180722T030541_N0206_R075_T49RFP_20180722T060550.SAFE
                                                                          /AUX_DATA...
'''
os.system("python fuseimg.py")