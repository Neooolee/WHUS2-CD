# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:54 2018

@author: Neoooli
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from gdaldiy import *

def conv2d(input_,output_dim,kernel_size=3,stride=2,padding="SAME",biased=True):
    return Conv2D(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,use_bias=biased)(input_)
def deconv2d(input_,output_dim,kernel_size=4,stride=2,padding="SAME",biased=True):
    return Conv2DTranspose(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,use_bias=biased)(input_)
def DSC(input_, output_dim,kernel_size=3, stride=1, padding="SAME",scale=1, biased=True):
    return SeparableConv2D(input_.shape[-1],kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,depth_multiplier=scale,use_bias=biased)(input_)
def MDSC(input_,output_dim,kernel_list=[3,5], stride=1, padding="SAME",scale=1,biased=True):
    depthoutput_list=[]
    for i in range(len(kernel_list)):
        depth_output=DepthwiseConv2D(kernel_size=kernel_list[i],strides=stride,padding=padding,depth_multiplier=scale)
        depthoutput_list.append(depth_output(input_))
    output = concatenate(depthoutput_list,axis=-1)
    output = conv2d(output,output_dim,kernel_size=1,stride=1,padding=padding,biased=biased)
    return output
def SDC(input_,output_dim, kernel_size=3,stride=1,dilation=2,padding='SAME', biased=True):
    """
    Smoothed dilated conv2d via the Separable and Shared Convolution (SSC) without BN or relu.
    """
    input_dim = input_.shape[-1]
    fix_w_size = dilation * 2 - 1
    eo = tf.expand_dims(input_,-1)
    o = Conv3D(1,kernel_size=[fix_w_size, fix_w_size,1],strides=[stride,stride,stride],padding=padding,use_bias=biased)(eo)
    o = eo + o
    o = tf.squeeze(o,-1)
    o = Conv2D(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,dilation_rate=(dilation, dilation),use_bias=biased)(o)
    return o
def SDRB(input_, kernel_size=3,stride=1,dilation=2,training=False, biased=True):
    output_dim=input_.get_shape()[-1]
    sconv1=SDC(input_,output_dim, kernel_size,stride,dilation,biased=biased)
    sconv1=batch_norm(sconv1,training)
    sconv1= relu(sconv1)
    sconv2=SDC(sconv1,output_dim, kernel_size,stride,dilation,biased=biased)
    sconv2=batch_norm(sconv2,training)
    return relu(sconv2+input_) 
def relu(input_):
    return ReLU()(input_)
def lrelu(input_):
    return LeakyReLU()(input_)
def avg_pooling(input_,kernel_size=2,stride=2,padding="same"):
    return tf.keras.layers.AveragePooling2D((kernel_size,kernel_size),stride,padding)(input_)
def max_pooling(input_,kernel_size=2,stride=2,padding="same"):
    return tf.keras.layers.MaxPool2D((kernel_size,kernel_size),stride,padding)(input_)
def dropout(input_,rate=0.2,training=True):
    """
    rate是丢掉多少神经元.
    """  
    return tf.keras.layers.Dropout(rate)(input_,training)
def GAP(input_):
    return GlobalAveragePooling2D()(input_)
def batch_norm(input_,training=True):
    return BatchNormalization()(input_,training)
class InstanceNormalization(tf.keras.layers.Layer):
  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def instance_norm(input_):
    return InstanceNormalization()(input_)

def norm(input_,norm='batch_norm',training=True):
    if norm==None:
        return input_
    elif norm=='batch_norm':
        return BatchNormalization()(input_,training)
    elif norm=='instance_norm':
        return InstanceNormalization()(input_)

def act(input_,activation='relu'):
    if activation==None:
        return input_
    elif activation=='relu':
        return ReLU()(input_)
    elif activation=='lrelu':
        return LeakyReLU(alpha=0.2)(input_)

  
def diydecay(steps,baselr,cycle_step=100000,decay_steps=100,decay_rate=0.96):
    n=steps//cycle_step
    clr=baselr*(0.8**n)   
    steps=steps-n*cycle_step
    k=steps//decay_steps
    i=(-1)**k
    step=((i+1)/2)*steps-i*((k+1)//2)*decay_steps
    dlr = clr*decay_rate**(int(step))      
    return dlr
def decay(global_steps,baselr,start_decay_step=100000,cycle_step=100000,decay_steps=100,decay_rate=0.96):
    lr=np.where(np.greater_equal(global_steps,start_decay_step),
                diydecay(global_steps-start_decay_step,baselr,cycle_step,decay_steps,decay_rate),
                baselr)
    return lr

def l2_loss(x):
    return tf.sqrt(tf.reduce_sum(x**2))
  
def randomflip(input_,n):
    #生成-3到2的随机整数，-1顺时针90度，-2顺时针180，-3顺时针270,0垂直翻转，1水平翻转，2不变
    if n<0:
        return np.rot90(input_,n)
    elif -1<n<2:
        return np.flip(input_,n)
    else: 
        return input_
def read_img(datapath,scale=255):
    img=imgread(datapath)/scale   
    return img
def read_imgs(datapath,scale=255,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=read_img(datapath[i],scale)
        img = randomflip(img,k)
        img=img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    return tf.convert_to_tensor(imgs,tf.float32)

def read_labels(datapath,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=imgread(datapath[i])
        img=randomflip(img,k)
        img=img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    imgs = rgb_to_gray(imgs) 
    return tf.convert_to_tensor(imgs,tf.float32)

rgb_colors=OrderedDict([
    ("cloud-free",np.array([0],dtype=np.uint8)),
    ("cloud",np.array([255],dtype=np.uint8))])

def rgb_to_gray(rgb_mask):
    label = (np.zeros(rgb_mask.shape[:3]+tuple([1]))).astype(np.uint8)
    if len(rgb_mask.shape)==4:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((rgb_mask == np.asarray(rgb_values)).astype(np.uint8).sum(-1) == 3)
            label[match_pixs] = gray        
    else:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((rgb_mask == np.asarray(rgb_values)).astype(np.uint8) == 1)
            label[match_pixs] = gray
    return label.astype(np.uint8)

#输入shape=(w,h,c)/(batch_size,w,h,c)
def label_to_rgb(labels):
    max_index=np.argmax(labels,axis=-1)#第三维上最大值的索引，返回其他维度，并在并对位置填上最大值之索引
    n=len(labels.shape)-1
    if labels.shape[-1]<3:
        rgb = (np.zeros(labels.shape[:n])).astype(np.uint8)
    else:
        rgb = (np.zeros(labels.shape[:n]+tuple([3]))).astype(np.uint8)
    for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
        match_pixs = np.where(max_index == gray)    
        rgb[match_pixs] = rgb_values
    return rgb.astype(np.uint8)

def down_sample(input_,kernel_size,classes):
    onehot=tf.one_hot(tf.cast(input_,dtype=tf.int32),classes)
    onehot=avg_pooling(onehot,kernel_size,kernel_size)
    onehot=tf.argmax(onehot,axis=-1)
    return onehot