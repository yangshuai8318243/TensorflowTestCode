# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:45:06 2019

@author: ShuaiYang
"""

import tensorflow as tf
#创建常量
m1 = tf.constant([[3,3]])

m2 = tf.constant([[2],[3]])

product = tf.matmul(m1,m2) 
print(m1)
print(m2)
with  tf.Session() as sess:
#sess = tf.Session()
    data = sess.run(product)
    print(data)# -*- coding: utf-8 -*-

