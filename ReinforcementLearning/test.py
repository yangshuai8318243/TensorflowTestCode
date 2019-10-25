# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-25 16:17:28
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-10-25 16:40:06
import pandas as pd
import numpy as np
import tkinter as tk
import tensorflow as tf


table = pd.DataFrame(
	np.zeros((5 , 2)),
	columns = ['a','b'] ,
)


a = 0 if False else 5

# tab = tf.squared_difference(11, 13)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# print(tab)

index = np.random.choice(5, size = 5)
ind = 5
memory = np.array([[1, 2], [2, 21],[3, 22], [4, 23],[5, 24]])
memory1 = np.array([[1,2,3,4,5,6]])
batch_memory = memory[0,1]

# batch_memory[:,:]
test = np.arange(200 , dtype = np.int32)
print(test)
print("=======111============")

print(np.argmax(memory))
print(np.argmax(memory , axis = 1))
print("=======222============")

print(batch_memory)
print("=========333==========")
test1 = batch_memory[:,1].astype(int)
print(test1)

# print(table)
# table.loc[2,'b'] += 1
# print()
# print(table)
# table.loc[2,'b'] += 1

# print(table)

# strDat = '{0}x{1}'
# st = strDat.format(50 * 4, 50 * 5)
# print(strDat)
# print(st)

# for x in range(0,50 * 4, 50):
# 	print(x)

# 	pass

# UNIT = 50 #每个格子所占的像素
# MAZE_H = 5 #环境的高度的格子数
# MAZE_W = 5 #环境的宽度的格子数
# origin = np.array([UNIT * 0.5,UNIT*0.5])shu

# isMax = state_action == np.max(origin)

# obj = tk.Tk()
# obj.title("Box")
# obj.geometry('{0}x{1}'.format(50*5,50*5))
# canvas = tk.Canvas(obj,bg = "white",
# 								height = MAZE_H * UNIT,
# 								width = MAZE_W * UNIT) #定义画布

# rect = canvas.create_rectangle(
#             0, 0,
#              30,  100,
#             fill='red')

# canvas.pack()

# pos = canvas.coords(rect)

# print("------->",pos)

# obj.mainloop()

# tablePos = pd.DataFrame(
# 			np.zeros((MAZE_W , MAZE_H)).astype(np.str),
# 		)
# posArr = pd.DataFrame(
# 				np.zeros((2,3)),
# 			)


# print(tablePos)

# listD = "225.0-25.0".split('-')
# listD = ["sda","sd222"]
# print(listD.index("sss"))
# print()
# actions = ["a","b","c"]
# saveData = pd.DataFrame(columns = actions)
# print(saveData)
# listData = ["paht","",""]
# index = 1
# saveData = saveData.append(
# 	pd.Series(
# 		listData,
# 		#位置在末尾
# 		index = saveData.columns,
# 		name = 1,
# 		)

# 	)

# item = saveData.loc[1,:]
# item["b"] = "testt"
# print("--------------")
# print(index++)

# print("--------------")
# print(saveData)