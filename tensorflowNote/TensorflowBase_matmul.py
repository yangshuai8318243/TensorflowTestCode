# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-24 17:57:23
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-24 18:46:16
import tensorflow as tf


# 2-D tensor `a`
a = tf.constant([1, 2, 3, 4, 5, 6],shape=[2,3])
print("--a-",a)

# 2-D tensor `b`
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
print("--b-",b)

# 3-D tensor `a`
c = tf.matmul(a, b)
print("--c---",c)

with tf.Session() as sess:
	print("------a------")
	print(sess.run(a))
	print("-----b-------")
	print(sess.run(b))
	print("------c------")
	print(sess.run(c))