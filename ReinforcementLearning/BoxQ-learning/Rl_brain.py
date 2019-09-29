# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-27 16:50:51
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-09-29 16:23:41
import numpy as np
import pandas as pd


class QLearingTable:
	"""docstring for QLearingTable"""
	def __init__(self,actions,learning_rate = 0.01 , reward_decay = 0.9,e_greedy =0.9):
		super(QLearingTable, self).__init__()
		#存在的动作的数量
		self.actions = actions
		#学习率
		self.lr = learning_rate
		# 衰减度
		self.gamma = reward_decay
		# 贪婪度
		self.epsilon = e_greedy

		self.q_table = pd.DataFrame(columns = self.actions)

	# 选择动作
	def choose_action(self,observation):
		#检查动作
		self.check_state_exist(observation)

		#判断是否贪婪
		if np.random.uniform() < self.epsilon:
			
			state_action = self.q_table.loc[observation,:]

			isMax = state_action == np.max(state_action)
			#取出最大值，有可能最大值相同，所以是个list
			index = state_action[isMax].index
            #随机取出最大值表中的一个动作
			action = np.random.choice(index)
		else:
			action = np.random.choice(self.actions)

		return action
		pass

	#参数 s 之前的状态。
	#参数 a 当前选择的行动
	#参数 r 当前选择行动后得到的反馈
	#参数 s_ 当前选择的状态
	def learn(self,s , a, r, s_):
		#检查这次选择的动作
		self.check_state_exist(s_)
		#从q表中取出上传状态，所对应的动作的数据
		q_predict = self.q_table.loc[s,a]
		#如果是结束状态，
		if s_ == "terminal":
			# 当前选择的价值  = 当前环境反馈的价值 + 衰减值 * 反馈预计下一步的价值
			q_target = r + self.gamma * self.q_table.loc[s_,:].max()
		else:
			q_target = q_predict;

		# 更新q表 更新值 = 原有价值 + 学习率* (当前选择价值 - 原有价值)
		self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

		pass

	#参数 state 当前状态
	def check_state_exist(self,state):
		#如果没有这个状态，则添加到q表中
		if state not in self.q_table.index:
			#创建一个，和动作长度相同的数组，插入到q表中
			listData = [0]*len(self.actions)
			self.q_table = self.q_table.append(
				pd.Series(
					listData,
					#位置在末尾
					index = self.q_table.columns,
					name = state,
					)

				)

		pass
		