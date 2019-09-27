# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-23 15:50:54
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-09-25 16:31:40
# import tensorflow as tf
import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6 # 初始状态，初始长度
ACTIONS = ["left","right"] #可以执行的动作
EPSILON = 0.9 # 贪婪值，决定是否随机动作
ALPHA = 0.1 # 学习率
LAMBDA = 0.9 # 对于之后动作的使用的衰减值
MAX_EPISODES = 13 #最大回合数
FRESH_TIME = 0.3 # 每步时长
#创建一个 q表
def build_q_table(n_states , actions):
	# 创建一个 n_states列 len(actions) 行的表
	# 每行的名字为 actions
	table = pd.DataFrame(
		np.zeros((n_states , len(actions))),
		columns = actions,
	)
	print(table)
	return table;

#选择当前要做的行为 state 当前的状态  q_table 行为表
def choose_action(state , q_table):
	#获取当前状态的动作价值表   pd中的iloc方法获取对应key值的行，或者列，这里是行
	state_action = q_table.iloc[state,:]
	# 随机数大于贪婪值，或者当前行没有存储数据，则随机一个步骤
	if (np.random.uniform() > EPSILON) or (state_action.all() == 0):
		action_name = np.random.choice(ACTIONS)
	# 否则选择当前价值表中，较大的值
	else:
		action_name = state_action.idxmax()
	return  action_name

# 环境对行为的反馈 S当前的状态  A 当前行为
# S_ 环境处理后当前所在的状态，R 当前所在的价值
def get_env_feedback(S , A):
	if A == "right":
		# -2 是因为，本身小人算一格，同时终点算一格，所以当当达到-2时就是终点
		if S == N_STATES -2:
			S_ = "terminal"
			R = 1
		else:
			S_ = S+1
			R= 0
	else:
		R = 0
		if S == 0:
			S_ = S 
		else:
			S_ = S -1
	return S_ , R
	pass

#环境建立
# S 当前所在位置 episode 当前训练次数   step_counter 本次训练所用步数
def update_env(S, episode, step_counter):
    # 创建环境
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' 
    #如果到达终点则打印本次训练结果
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    #否则打印当前位置
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


# 创建训练循环
def rl():
	#初始化 q table
	q_table = build_q_table(N_STATES,ACTIONS)
	#循环训练
	for episode in range(MAX_EPISODES):
		#记录当前训练使用步数
		step_counter = 0
		#初始化位置
		S = 0
		# 是否停止标示
		is_terminated = False
		#初始化当前训练环境
		update_env(S,episode,step_counter)

		while not is_terminated:
			#选择下一步的动作
			A = choose_action(S,q_table)
			#计算环境对当前动作的反馈
			S_ ,R = get_env_feedback(S,A)
			#取出当前状态，动作的价值
			print("\n----------------")
			print("S A -- ",S,A)
			print("----------------")

			q_predict = q_table.loc[S,A]
			#如果没有结束
			if S_ != "terminal":
				# 当前选择的价值  = 当前环境反馈的价值 + 衰减值 * 反馈预计下一步的价值
				q_target = R + LAMBDA * q_table.iloc[S_,:].max()
			#如果到达终点
			else:
				q_target = R
				is_terminated = True

			# 更新q表 更新值 = 原有价值 + 学习率* (当前选择价值 - 原有价值)
	
			q_table.loc[S , A] += ALPHA * (q_target - q_predict)
			#更新下一步的位置
			S = S_

			#更新环境中当前位置
			update_env(S,episode,step_counter + 1)
			#叠加当前训练步数
			step_counter += 1

			pass

		pass
	return q_table;
	pass

if __name__ == "__main__":
	q_table = rl()

	print('\r\n Q-table:\n')
	print(q_table)