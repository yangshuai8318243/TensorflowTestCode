# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-04-16 19:42:04
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-04-16 20:43:51

import tensorflow as tf;
# 向前传播算法 矩阵
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)