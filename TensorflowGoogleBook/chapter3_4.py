# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-16 19:42:04
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-16 20:43:51

import tensorflow as tf;
# 向前传播算法 矩阵
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
print("--------------------------weights------------------------------")
# 声明 2*3 的矩阵变量  疑问，正太分布是什么，
weights = tf.Variable(tf.random_normal([2,3],stddev = 2))
biases = tf.Variable(tf.zeros([3]))
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value()*2.0)
# 三种初始化的方式
sess = tf.Session()
with sess.as_default():
	tf.global_variables_initializer().run()
	print(sess.run(weights))
	print(sess.run(biases))
	print(sess.run(w2))
	print(sess.run(w3))