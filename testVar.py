# -*- coding: utf-8 -*-
# @Author: ShuaiYang
# @Date:   2019-04-02 19:05:45
# @Last Modified by:   ShuaiYang
# @Last Modified time: 2019-04-02 19:05:47
# -*- coding: utf-8 -*-
# @Author: ShuaiYang
# @Date:   2019-04-02 16:57:49
# @Last Modified by:   ShuaiYang
# @Last Modified time: 2019-04-02 19:05:20

import tensorflow as tf
 


# 首先，创建一个TensorFlow常量=>2
const = tf.constant(2.0, name='const')

# 创建TensorFlow变量b和c
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, dtype=tf.float32, name='c')

# 创建operation
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# 1. 定义init operation
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	# 2. 运行init operation
	sess.run(init_op)
	# 计算
	a_out = sess.run(a)
	print("Variable a is {}".format(a_out))

