# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-16 19:42:04
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-16 20:43:51

import tensorflow as tf;
# 声明变量w1 w2 这里通过seed参数设定随机种子
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

# 暂时将输入的特征向量定义为常量，注意x是一个1*2的矩阵
x = tf.constant([[0.7,0.9]])

# 向前传播算法获得神经网络输出
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
with sess.as_default():
	sess.run(w1.initializer)
	sess.run(w2.initializer)
	print(sess.run(y))