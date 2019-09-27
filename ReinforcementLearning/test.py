# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-25 16:17:28
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-09-27 18:14:34
import pandas as pd
import numpy as np
import tkinter as tk


table = pd.DataFrame(
	np.zeros((5 , 2)),
	columns = ['a','b'] ,
)

print(table)
table.loc[2,'b'] += 1
print()
print(table)
table.loc[2,'b'] += 1

print(table)

strDat = '{0}x{1}'
st = strDat.format(50 * 4, 50 * 5)
print(strDat)
print(st)

for x in range(0,50 * 4, 50):
	print(x)

	pass

UNIT = 50 #每个格子所占的像素
MAZE_H = 5 #环境的高度的格子数
MAZE_W = 5 #环境的宽度的格子数
origin = np.array([UNIT * 0.5,UNIT*0.5])

isMax = state_action == np.max(origin)

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

# print(listD)
# print()