# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-11 19:09:13
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-11 20:01:54
import tensorflow as tf;

print("-----------------通过计算机图 对数据进行隔离计算---------------------------")
g1 = tf.Graph()
with g1.as_default():
	v = tf.get_variable("v",initializer = tf.zeros_initializer() ,shape = 1 )

g2 = tf.Graph()
with g2.as_default():
	v = tf.get_variable("v",initializer = tf.ones_initializer() ,shape = 1 )

with tf.Session(graph = g1) as sess:
	tf.global_variables_initializer().run()

	with tf.variable_scope("",reuse = True):
		print(sess.run(tf.get_variable("v")))


with tf.Session(graph = g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("",reuse = True):
		print(sess.run(tf.get_variable("v")))
# 疑问，如何具体的使用计算图，计算图如何在当前图中进行计算，如何做到关联当前图中的相关变量

print("----------------------------张量学习----------------------------------")

a = tf.constant([1.0,3.0], name='a')
b = tf.constant([3.0, 5.2], name='b')
result = tf.add(a, b)

print("-->",result)

print("-------------------------Session 会话------------------------------------")

# 使用session的过程中必须使用 with tf.Session() 的形式，tf使用python的上下文管理来管理会话，只要在with中就不需要调用 Session.close()

