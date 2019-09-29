# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-25 17:41:45
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-09-29 16:57:47


import numpy as np
import pandas as pd
import time
import sys
# 判断当前python版本
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import random


UNIT = 50 #每个格子所占的像素
MAZE_H = 5 #环境的高度的格子数
MAZE_W = 5 #环境的宽度的格子数
#起点位置
origin = np.array([UNIT * 0.5,UNIT*0.5])

class Box(tk.Tk,object):
	"""docstring for Box"""
	def __init__(self):
		super(Box, self).__init__()
		self.action_space = {"u","d","l","r"} #可以做的操作
		self.n_actions = len(self.action_space) #可以操作的数量
		self.title("Box")
		self.geometry('{0}x{1}'.format(MAZE_H*UNIT,MAZE_W*UNIT)) #设置窗口大小
		self._initTabPos()
		self._build_box()


	def _build_box(self):
		self.canvas = tk.Canvas(self,bg = "white",
								height = MAZE_H * UNIT,
								width = MAZE_W * UNIT) #定义画布

		#创建格子
		#纵线
		for c in range(0, MAZE_W * UNIT, UNIT):
			x0 , y0 , x1 , y1 = c, 0, c, MAZE_H * UNIT;
			self.canvas.create_line(x0,y0,x1,y1)
			pass
		#横线
		for r in range(0, MAZE_H * UNIT, UNIT):
			x0 , y0 , x1 , y1 = 0 , r , MAZE_W * UNIT , r;
			self.canvas.create_line(x0,y0,x1,y1)
			pass

		# 计算随机位置
		posArr = self._random_pos()
		#障碍物1
		hell1_center = posArr.loc[0,:] 

		print(hell1_center[0],hell1_center[1])		

		self.hell1 = self.canvas.create_rectangle(
			hell1_center[0] - 15, hell1_center[1] - 15,
			hell1_center[0] + 15, hell1_center[1] + 15,
			fill = "black")	

		#障碍物2
		hell2_center = posArr.loc[1,:]
		self.hell2 = self.canvas.create_rectangle(
			hell2_center[0] - 15, hell2_center[1] - 15,
			hell2_center[0] + 15, hell2_center[1] + 15,
			fill = "black")	


		# 终点
		oval_center = posArr.loc[2,:] 
		self.oval = self.canvas.create_oval(
			oval_center[0] - 15, oval_center[1] - 15,
			oval_center[0] + 15, oval_center[1] + 15,
			fill='yellow')

		# 玩家
		self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
		
		# 绘制模块
		self.canvas.pack()

		pass
	def _initTabPos(self):
		self.tablePos = pd.DataFrame(
				np.zeros((MAZE_W , MAZE_H)).astype(np.str),
			)
		for x in range(MAZE_W):
			for y in range(MAZE_H):
				posX = (x+1) * UNIT - UNIT*0.5;
				posY = (y+1) * UNIT - UNIT*0.5;
				self.tablePos.loc[x,y] = str(posX) + "-" +  str(posY)
				pass
			pass
		pass
	#计算物体随机位置
	def _random_pos(self):
			
		posArr = pd.DataFrame(
				np.zeros((2,1)),
			)

		x,y = self._random_index();
		
		raPos1 = self.tablePos.loc[x,y]
		strlist = raPos1.split('-')
		
		posArr.loc[0,0] = float(strlist[0])
		posArr.loc[0,1] = float(strlist[1])

		while True:
			x,y = self._random_index();
			raPos2 = self.tablePos.loc[x,y]
			if raPos1 != raPos2:
				strlist = raPos2.split('-')
				posArr.loc[1,0] = float(strlist[0])
				posArr.loc[1,1] = float(strlist[1])
				break
			pass

		while True:
			x,y = self._random_index();
			raPos3 = self.tablePos.loc[x,y]
			if raPos1 != raPos3:
				strlist = raPos3.split('-')
				posArr.loc[2,0] = float(strlist[0])
				posArr.loc[2,1] = float(strlist[1])
				break
			pass



		return posArr;


	def _random_index(self):
		x = random.randint(0,4)
		y = random.randint(0,4)
		return x, y
		pass

	# 重启
	def reset(self):
		#刷新view
		self:update()
		time.sleep(0.5)
		# 删除原有 玩家
		self.canvas.delete(self.rect)

		# 玩家
		self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

		posArr = [self.canvas.coords(self.hell1),self.canvas.coords(self.hell2)];

		self.canvas.delete(self.hell1)
		self.hell1 = self.canvas.create_rectangle(
			posArr[0][0], posArr[0][1],
			posArr[0][2], posArr[0][3],
			fill = "black")

		self.canvas.delete(self.hell2)
		self.hell2 = self.canvas.create_rectangle(
			posArr[1][0], posArr[1][1],
			posArr[1][2], posArr[1][3],
			fill = "black")

		# 重新设置item的坐标
		# 返回对应坐标
		return self.canvas.coords(self.rect)
		pass

	# 移动一步
	def step(self, acction):
		pos = self.canvas.coords(self.rect)
		#初始化动作列表 -------> [5.0, 205.0, 35.0, 235.0]
		# 记录当前位置
		base_action = np.array([0,0])
		if acction == 0: # up
			if pos[1] > UNIT:
				base_action[1] -= UNIT
		elif acction == 1: # down
			if pos[1] < (MAZE_H -1) * UNIT :
				base_action[1] += UNIT
		elif acction == 2 : # right
			if pos[0] < (MAZE_W -1 ) * UNIT:
				base_action[0] += UNIT
		elif acction == 3 : #left
			if pos[0] > UNIT:
				base_action[0] -= UNIT
		# 绘制玩家当前位置
		self.canvas.move(self.rect, base_action[0] , base_action[1])

		# 确认玩家在当前画布中的位置
		pos_ = self.canvas.coords(self.rect)
		posArr = [self.canvas.coords(self.hell1),self.canvas.coords(self.hell2)];
		#判断玩家是否接触到其他物体
		if pos_ == self.canvas.coords(self.oval):
			reward = 1
			done = True
			state = "terminal"
		# 判断某个表或者变量是否在这个数组中
		elif pos_  in posArr:
			reward = -1
			done = True
			state = "terminal"
			hallIndex = posArr.index(pos_)
			hallPos = posArr[hallIndex]
			if hallIndex == 0 :
				self.canvas.delete(self.hell1)
				self.hell1 = self.canvas.create_rectangle(
					hallPos[0], hallPos[1],
					hallPos[2], hallPos[3],
					fill = "blue")

			elif hallIndex == 1:
				self.canvas.delete(self.hell2)
				self.hell2 = self.canvas.create_rectangle(
					hallPos[0], hallPos[1],
					hallPos[2], hallPos[3],
					fill = "blue")

		else:
			reward = 0
			done = False
			state = pos_
		pass

		return state , reward , done

	def render(self):
		self.update()
		time.sleep(0.1)
		
		pass


def update():
	for x in range(3):
		s = env.reset()
		while True:
			env.render()
			a = 1
			s,r,d = env.step(a)
			if d:
				break
			pass
		pass

	pass


if __name__ == '__main__':
    env = Box()
    env.after(100, update)
    env.mainloop()