# -*- coding: utf-8 -*-
# @Author: ShuaiYang
# @Date:   2019-04-02 16:57:49
# @Last Modified by:   ShuaiYang
# @Last Modified time: 2019-04-08 19:06:15
# https://blog.csdn.net/dufufd/article/details/78931840
import tensorflow as tf

# 创建变量 W 和 b 节点，并设置初始值
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W*x + b
 
# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)
# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))
 
# 创建 Session 用来计算模型
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run(W))
print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))


# 给 W 和 b 赋新值
fixW = tf.assign(W, [2.])
fixb = tf.assign(b, [1.])
# run 之后新值才会生效
sess.run([fixW, fixb])
# 重新验证损失值
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))