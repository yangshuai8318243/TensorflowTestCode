# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-05-15 14:47:16
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-05-17 16:40:34


import tensorflow as tf
import numpy as np
#数据准备 使用NumPy 生成假数据

x_data = np.float32(np.random.rand(2,3,3)) #随机输入
# y_data = np.dot([0.100,0.200],x_data) + 0.3

print("x_data  : ",x_data)
# print("y_data  : ",y_data)