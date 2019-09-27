# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-27 16:50:51
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-09-27 18:25:12
import numpy as np
import pandas as pd


class QLearingTable():
	"""docstring for QLearingTable"""
	def __init__(self,actions,learning_rate = 0.01 , reward_decay = 0.9,e_greedy):
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
			
			state_action = self.q_table[observation,:]

			isMax = state_action == np.max(state_action)
			#取出最大值，有可能最大值相同，所以是个list
            index = state_action[isMax].index
            #随机取出最大值表中的一个动作
			action = np.random.choice(index)
		else
			np.random.choice(self.actions)

		return action
		pass

	#参数 s 之前的状态。
	#参数 a 当前选择的行动
	#参数 r 当前选择行动后得到的反馈
	#参数 s_ 当前选择的状态
	def learn(self,s , a, r, s_):
		pass

	#参数 state 当前状态
	def check_state_exist(self,state):
		pass
		