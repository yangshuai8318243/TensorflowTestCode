# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:02:46 2019

@author: ShuaiYang
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
# print(sess.run(hello))

# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0)# also tf.float32 implicitly
# print(sess.run([node1, node2]))

# node3 = tf.add(node1, node2)
# print("node3:", node3)
# print("sess.run(node3):", sess.run(node3))

# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# print("adder_node:", adder_node)
# # print("adder_node sess.run:", sess.run(adder_node))
# print(sess.run(adder_node, {a:3, b:4.5}))
# print(sess.run(adder_node, {a: [1,3], b: [2,4]}))


# add_and_triple = adder_node *3
# print(sess.run(add_and_triple, {a:3, b:4.5}))

# 创建一个整型常量，即 0 阶 Tensor
t0 = tf.constant(3, dtype=tf.int32)
 
# 创建一个浮点数的一维数组，即 1 阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)
 
# 创建一个字符串的2x2数组，即 2 阶 Tensor
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)
 
# 创建一个 2x3x1 数组，即 3 阶张量，数据类型默认为整型
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])
 
# 打印上面创建的几个 Tensor
print(t0)
print(t1)
print(t2)
print(t3)
print("=============")

print(sess.run(t0))
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t3))
