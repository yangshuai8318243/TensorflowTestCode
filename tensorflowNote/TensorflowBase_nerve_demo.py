# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-24 17:44:03
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-29 18:25:29
# 训练的 函数 y_data = x_data * 0.1 + 0.3
# https://www.jianshu.com/p/596a30d46f34
import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



x_data = np.linspace(-1,1,300)[:,np.newaxis]
print("---np.newaxis--->",np.newaxis)
print("---x_data--->",x_data)
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
print("---np.square(x_data)--->",np.square(x_data))
print("---x_data.shape--->",x_data.shape)
print("---tf.nn.relu--->",tf.nn.relu)
# print("---noise--->",noise)

#None表示给多少个sample都可以
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))