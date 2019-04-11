# -*- coding: utf-8 -*-
# @Author: ShuaiYang
# @Date:   2019-04-02 16:57:49
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-09 19:21:52
# https://blog.csdn.net/dufufd/article/details/78931840
import tensorflow as tf
print("true data ->>",4.8, 8.5, 10.4, 21.0, 25.3)

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
print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))


# 给 W 和 b 赋新值
fixW = tf.assign(W, [2.])
fixb = tf.assign(b, [1.])
# run 之后新值才会生效
sess.run([fixW, fixb])
# 重新验证损失值
print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))

print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
 
# 用两个数组保存训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]
 
# 训练10000次
# for i in range(10000):
#     sess.run(train, {x: x_train, y: y_train})
 
# # 打印一下训练后的结果
# print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: x_train , y: y_train})))

# # 重新验证损失值
# print("重新验证损失值 - >")
# print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))