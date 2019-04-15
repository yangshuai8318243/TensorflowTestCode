# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-15 19:07:27
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-15 19:24:55

import tensorflow as tf;




print("-------------------Session-------------------------")
# 使用session的过程中必须使用 with tf.Session() 的形式，tf使用python的上下文管理来管理会话，只要在with中就不需要调用 Session.close()
a = tf.constant([1.0,3.0], name='a')
b = tf.constant([3.0, 5.2], name='b')
result = tf.add(a, b)

sess = tf.Session()
# 方式1
print(result.eval(session = sess))
# 方式2
with sess.as_default():
	print(result.eval())

print("-------------------ConfigProto-------------------------")
# 1、记录设备指派情况 :  tf.ConfigProto(log_device_placement=True)
# 设置tf.ConfigProto()中参数log_device_placement = True ,
# 可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。
# 
# 2. 自动选择运行设备 ： tf.ConfigProto(allow_soft_placement=True)
# 在tf中，通过命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备。如果手动设置的设备不存在或者不可用，
# 就会导致tf程序等待或异常，为了防止这种情况，可以设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。

config = tf.ConfigProto(allow_soft_placement = True,log_device_placement = True);
sess1 = tf.InteractiveSession(config = config)
sess2 = tf.Session(config = config)
print(result.eval(session = sess2))

print("-------------------ConfigProto---------per_process_gpu_memory_fraction----------------")

# 3. 限制GPU资源使用
# tf提供了两种控制GPU资源使用的方法，
# 一是让TensorFlow在运行过程中动态申请显存，需要多少就申请多少;第二种方式就是限制GPU的使用率。
# 

# 一、动态申请显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session1 = tf.Session(config=config)
# 二、限制GPU使用率

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
session1 = tf.Session(config=config)
print(result.eval(session = session1))

print("-------------------ConfigProto---------GPU----------------")
# 设置使用哪块GPU
# 方法一、在python程序中设置：
# 在执行python程序时候：
# CUDA_VISIBLE_DEVICES=0,1 python yourcode.py


sess33 = tf.Session()
print(result.eval(session = sess33))
