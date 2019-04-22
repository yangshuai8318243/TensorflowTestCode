# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-22 19:45:54
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-22 20:00:37
# 
import tensorflow as tf;

# 声明变量w1 w2 这里通过seed参数设定随机种子
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

#定义 placeholder 作为存放数据的地方，这里的维度不需要定义
#如果维度确定，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape = (3,2) ,name = "inpout" )
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
with sess.as_default():
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	cross_entropy = tf.reduce_mean(y_* tf.log(tf.clip_by_value(y,le-10,1.0)))
	# print(sess.run(y,feed_dict = {x:[[0.7,0.9]]}))
	print(sess.run(y,feed_dict = {x:[[0.7,0.9],[0.1,0.3],[0.5,0.9]]}))