# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:57:36 2018

@author: Neoooli
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import tensorflow as tf

import numpy as np
import glob
from PIL import Image
from collections import OrderedDict
from CDFM3SF import *
from ops import *
from gdaldiy import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser(description='')
parser.add_argument("--snapshot_dir", default='./snapshots/', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=[384,192,64], help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate for adam') #基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch') #训练的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=100, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=1000, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=10000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--x_train_data_path", default='f:/lijun/data/graduatedata/clouddetection/S2A/test/trainDNclips/10m', help="path of x training datas.") #x域的训练图片路径
parser.add_argument("--y_train_data_path", default='f:/lijun/data/graduatedata/clouddetection/S2A/test/trainlabelclips/10m', help="path of y training datas.") #y域的训练图片路径
parser.add_argument("--batch_size", type=int, default=16,help="load batch size") #batch_size
parser.add_argument("--bands", type=int, default=[4,6,3], help="load batch size") #batch_size
parser.add_argument("--classes", type=int, default=1, help="load batch size")
parser.add_argument("--output_level", type=int, default=1, help="load batch size")
args = parser.parse_args()

rgb_colors=OrderedDict([
    ("cloud-free",np.array([0],dtype=np.uint8)),
    ("cloud",np.array([255],dtype=np.uint8))])  
def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #保存的模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
   if not os.path.exists(logdir): #如果路径不存在即创建
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')

def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img+1)*127.5
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像
def acv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = img*255.0
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像 
def get_write_picture(batch_x_image,batch_label_image,pre): #get_write_picture函数得到训练过程中的可视化结果
    batch_x_image=batch_x_image[0][:,:,[0,1,2]]
    low,high=np.percentile(batch_x_image,(2,98))
    batch_x_image[low>batch_x_image]=low
    batch_x_image[batch_x_image>high]=high   
    rescaled_img=(batch_x_image-low)/(high-low)
    x_image = acv_inv_proc(rescaled_img) #还原x域的图像

    label = batch_label_image[0]*255 #还原x域的图像
    pre = acv_inv_proc(pre[0]>0.5)#还原y域的图像
    row1 = np.concatenate((x_image,np.concatenate((label,label,label),axis=2),np.concatenate((pre,pre,pre),axis=2)), axis=1) #得到训练中可视化结果的第一行
    return row1 
 
def make_train_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) #将x域图像数量与y域图像数量对齐
    return image_path_lists
    
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(tf.cast(src,tf.float32) - tf.cast(dst,tf.float32)))
def l2_loss(x):
    return tf.sqrt(tf.reduce_sum(x**2))
def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return tf.reduce_mean((tf.cast(src,tf.float32) - tf.cast(dst,tf.float32))**2)
class maintrain(object):
    """docstring for maintrain"""
    def __init__(self):
        super(maintrain, self).__init__() 
        self.Net = CDFM3SF(args.bands,training=True,name="CD-FM3SF")
        self.ag_optimizer = tf.keras.optimizers.Adam(args.base_lr,args.beta1,args.beta2)
        self.ckpt = tf.train.Checkpoint(Net=self.Net)

    @tf.function                
    def train_step(self,image_list,label,lr):
        self.ag_optimizer.lr.assign(lr)
        with tf.GradientTape(persistent=True) as a_tape:
            logits=self.Net(image_list)
            loss_list=[]
            loss_w_list=[1,0.1,0.01]
            cost_sum=0
            labels=label
            down_list=[1,2,6]
            for i in range(len(image_list)):
                if i>0:
                    labels = tf.expand_dims(down_sample(label[:,:,:,0],down_list[i],2),-1)
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.float32),logits=logits[i]))
                loss_list.append(cost)
                cost_sum=cost_sum+cost*loss_w_list[i]   
        grads_ag=a_tape.gradient(cost_sum,self.Net.trainable_variables)
        self.ag_optimizer.apply_gradients(zip(grads_ag,self.Net.trainable_variables))

        return cost_sum,tf.nn.sigmoid(logits[0])

    def train(self,x_datalists,y_datalists):
        print ('Start Training')
        #存储训练日志
        train_summary_writer = tf.summary.create_file_writer(args.snapshot_dir)
        ckpt_manager = tf.train.CheckpointManager(self.ckpt,args.snapshot_dir, max_to_keep=100)
        if ckpt_manager.latest_checkpoint:
            self.ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
            path=ckpt_manager.latest_checkpoint
            step=int(path.split('-')[-1])
        else: 
            step=1       
        leny=len(y_datalists)
        start_epoch=(step*args.batch_size)//leny+1
        start=(step-(start_epoch-1)*(leny//args.batch_size))*args.batch_size
        for epoch in range(start_epoch,args.epoch): #训练epoch数       
               #每训练一个epoch，就打乱一下x域图像顺序
            shuffle(y_datalists) #每训练一个epoch，就打乱一下y域图像顺序           
            data_list= [name.replace('labelclips','DNclips') for name in y_datalists]
            data_list1= [name.replace('10m','20m') for name in data_list]
            data_list2= [name.replace('10m','60m') for name in data_list]           
            while (start+args.batch_size)<leny:   
                k = np.random.randint(low=-3, high=3)             
                batch_input_img=read_imgs(data_list[start:start+args.batch_size],10000,k)
                batch_input_img1=read_imgs(data_list1[start:start+args.batch_size],10000,k)
                batch_input_img2=read_imgs(data_list2[start:start+args.batch_size],10000,k)
                batch_input_label=read_labels(y_datalists[start:start+args.batch_size],k) 
                lr=tf.convert_to_tensor(decay(step,args.base_lr),tf.float32)              
                l,logitss= self.train_step([batch_input_img,batch_input_img1,batch_input_img2],
                            batch_input_label,
                            lr) #得到每个step中的生成器和判别器loss
                step=step+1
                start=start+args.batch_size           
                if step% args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss',l.numpy(),step)
                if step% args.save_pred_every == 0: #每过summary_pred_every次保存训练日志
                    ckpt_manager.save(checkpoint_number=step)
                if step % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                    write_image = get_write_picture(batch_input_img.numpy(),
                            batch_input_label.numpy(),
                            logitss.numpy()) #得到训练的可视化结果
                    write_image_name = args.out_dir + "/out"+ str(epoch)+'_'+str(step)+ ".png" #待保存的训练可视化结果路径与名称
                    imgwrite(write_image_name,np.uint8(write_image)) #保存训练的可视化结果
                    print(str(epoch),str(step),l.numpy())
            start=0
            if epoch==40:
                ckpt_manager.save(checkpoint_number=epoch)
                exit()

def main():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')#获取GPU列表
    tf.config.experimental.set_memory_growth(gpus[0], True)#设置GPU动态申请)])
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    x_datalists = make_train_data_list(args.x_train_data_path) #得到数量相同的x域和y域图像路径名称列表
    y_datalists = make_train_data_list(args.y_train_data_path)
    
    maintrain_object=maintrain()
    maintrain_object.train(x_datalists,y_datalists)
                
if __name__ == '__main__':
    main()

