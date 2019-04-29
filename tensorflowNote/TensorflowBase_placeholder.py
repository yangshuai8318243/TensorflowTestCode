# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-24 17:08:37
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-24 17:08:43
import tensorflow as tf

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3.],input2:[5]}))