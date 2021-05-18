# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:56:44 2018

@author: lijun
"""
import tensorflow as tf
from ops import *
def CDFM3SF(input_dim, gf_dim=64, reuse=False,training=False, name="CD-FM3SF"):
    # dropout_rate = 0.8
    input_ = tf.keras.layers.Input(shape=[None,None,input_dim[0]]) 
    input_1 = tf.keras.layers.Input(shape=[None,None,input_dim[1]])
    input_2 = tf.keras.layers.Input(shape=[None,None,input_dim[2]])
    e10 = relu(conv2d(input_,gf_dim,kernel_size=3,stride=1))
    e1 = relu(batch_norm(MDSC(e10,gf_dim,stride=1),training))
    e1 = e10+e1
        # e1 is (128 x 128 x self.gf_dim)
    p1=max_pooling(e1,2,2)
    e20 = relu(conv2d(input_1,gf_dim,kernel_size=3,stride=1))        
    c120 = tf.concat([p1,e20],axis=-1)  
    e2 = relu(batch_norm(MDSC(c120,gf_dim,stride=1),training))
    e2 = p1+e20+e2
        # e2 is (64 x 64 x self.gf_dim*2)
    p2=max_pooling(e2,3,3)
    e30 = relu(conv2d(input_2,gf_dim,kernel_size=3,stride=1))
    c230 = tf.concat([p2,e30],axis=-1)               
    e3 = relu(batch_norm(MDSC(c230,gf_dim,stride=1),training))
    e3 = p2+e30+e3
        # e3 is (32 x 32 x self.gf_dim*4)
    p3 = max_pooling(e3,2,2)
    e4 = relu(batch_norm(MDSC(p3,gf_dim,stride=1),training))
        # e3 is (32 x 32 x self.gf_dim*4)
    e4= p3+e4
        # e3 = tf.concat([e13,e3],axis=-1)
        # e3 = e3 + s3

    r1=SDRB(e4,3,1,2,training)
    r2=SDRB(r1,3,1,2, training)
    r3=SDRB(r2,3,1,3,training)
    r4=SDRB(r3,3,1,3,training)
    r5=SDRB(r4,3,1,4,training)
    r6=SDRB(r5,3,1,4,training)
        # d3 = tf.nn.dropout(d3, dropout_rate)
    d1 = tf.concat([e4,r2,r4,r6], axis=-1)
    d1 = relu(batch_norm(DSC(d1, gf_dim*2, stride=1), training))
        # d3 is (32 x 32 x self.gf_dim*8*2)
    d1 = deconv2d(d1, gf_dim,stride=2)
        # d4 = tf.nn.dropout(d4, dropout_rate)
    d1 = tf.concat([d1, e3], 3)
    d1 = relu(batch_norm(DSC(d1, gf_dim, stride=1),training))
    output3 = conv2d(d1,1,stride=1)
        # d3 is (32 x 32 x self.gf_dim*8*2)
    d2 = deconv2d(d1, gf_dim,stride=3)
        # d4 = tf.nn.dropout(d4, dropout_rate)
    d2 = tf.concat([d2, e2], 3)
    d2 = relu(batch_norm(DSC(d2, gf_dim,stride=1), training))
        # d4 is (16 x 16 x self.gf_dim*8*2)
    output2 = conv2d(d2,1,stride=1)
    d3 = deconv2d(d2, gf_dim)
        # d5 = tf.nn.dropout(d5, dropout_rate)
    d3 = tf.concat([d3, e1],3)          
        # d5 is (32 x 32 x self.gf_dim*4*2)
    d3 =  relu(batch_norm(DSC(d3,gf_dim,stride=1),training))
    output1 = conv2d(d3,1,stride=1)
        # d8 is (256 x 256 x output_c_dim)
    return tf.keras.Model([input_,input_1,input_2],[output1,output2,output3],name=name)
# model = generator_unet([4,6,3])
# print(model.summary())