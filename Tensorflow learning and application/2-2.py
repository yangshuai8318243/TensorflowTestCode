# -*- coding: utf-8 -*-
import tensorflow as tf

x=tf.Variable([1,2])
a= tf.constant([3,3])
sub = tf.subtract(x,a)
