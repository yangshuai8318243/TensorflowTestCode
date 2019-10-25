# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-10-15 15:42:37
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-10-25 16:58:10

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
	"""docstring for DoubleDQN"""
	def __init__(self, 
		n_action, #行为的数量
		n_features, #特征数量，环境反馈
		learning_rate = 0.005, #学习率
		reward_decay = 0.9, # 衰减度
		e_greedy = 0.9, #贪婪度
		replace_target_iter = 200, # 神经网络更新间隔
		memory_size = 200, # 数据记录大小
		batch_size = 32, # 神经网络每次学习的数量
		e_greedy_increment = None, # 贪婪度减小
		output_graph = False,
		double_q = True,
		sess = None,
		):


		super(DoubleDQN, self).__init__()
		self.n_action = n_action
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		self.double_q = double_q

		self.learn_step_counter = 0 #神经网络学习计数器

		self.memory = numpy.zeros((self.memory_size,n_features *2 +2))

		# 构建神经网络
		self._build_net()

		if sess is None:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
		else:
			self.sess = sess

		# 生成计算图
		if output_graph:
			tf.summary.FileWriter("logs/",self.sess.graph)

		# 记录误差值
		self.cost_his = []

		# 将评估神经网络的参数替换到目标神经网络中
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

	def _build_net():

		#构建神经网络层
		def build_layers(s,#动作，需要学习的输入值
			c_name, #数据所属命名
			n_L1,# 中间层神经元数量
			w_initializer, #权重矩阵初始化方式  具体参考 ：https://blog.csdn.net/UESTC_C2_403/article/details/72327321
			b_initializer #偏置值矩阵初始化方式
			):

			with tf.variable_scope("l1"):
				w1 = tf.get_variable("w1" , [self.n_features , n_L1] , initializer = w_initializer , collections = c_name)
				b1 = tf.get_variable("b1" , [1 , n_L1] , initializer = b_initializer, collections = c_name)
				l1 = tf.nn.relu( tf.matmul( s, w1) + b1)

			pass

			with tf.variable_scope("l2"):
				w2 = tf.get_variable("w2" , [n_L1 , self.n_action] , initializer = w_initializer , collections = c_name)
				b2 = tf.get_variable("b2" , [1 , self.n_action] , initializer = b_initializer, collections = c_name)
				l2 = tf.nn.relu( tf.matmul( l1, w2) + b2)

			pass
			return l2
		# ==============================构建评估神经网络（一直训练的神经网络）
		self.s = tf.placeholder(tf.folat32 , [ None , self.n_features ], name = "s")
		self.q_target = tf.placeholder(tf.folat32 , [None , self.n_action ], name = "Q_target") # 实际结果值的对应变量

		with tf.variable_scope("eval_net"):
			c_name , n_L1 , w_initializer , b_initializer = ['eval_net_params' , tf.GraphKeys.GLOBAL_VARIABLES],\
				20,tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
			
			self.q_eval = build_layers(self.s, c_name, n_L1 , w_initializer ,b_initializer)

		with tf.variable_scope("loss"):
			# tf.squared_difference 差平方计算函数 计算两个张量的平方差
			# tf.reduce_mean 是降维度的函数，在没有指定参数的情况下，取矩阵所以值求和算平均值
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval) )
		with tf.variable_scope("train"):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		# ============================构建目标神经网络（阶段性更新的神经网络，但是实际选择使用的神经网络）
		# 实际输入参数
		self.s_ = tf.placeholder(tf.folat32 , [None , self.n_features], name = "s_")
		with tf.variable_scope("target_net"):
			c_name = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

			self.q_next = build_layers(self.s_,c_name,n_L1,w_initializer,b_initializer);

		pass

	# 记录过渡数据，数据大小为memory_size
	# s 当前状态
	# a 当前动作
	# r 环境回馈的价值
	# s_ 下一步的状态
	def store_transition(self, s , a, r, s_):

		if not hasattr(self,"memory_counter"):
			self.memory_counter = 0

		# 横向合并表，将多个数据合并成一个数组
		transition = np.hstack((s , [a , r ],s_))
		# 每隔固定的memory_size 替换一次数据，通过index的方式获取当前需要记录数据的位置
		# 将数据 transition 放入 self.memory 中
		index = self.memory_counter % self.memory_size
		self.memory[index,:] = transition
		self.memory_counter += 1

		pass

	# 选择所需动作 
	def choose_action(self , observation):
		# 处理数据 增加维度 如：[1 2 3] 变为 [[1 2 3]]
		observation = observation[np.newaxis , :]
		# 使用评估神经网络进行学习，输出选项
		action_value = self.sess.run(self.q_eval , feed_dict = {self.s: observation} )
		# 选取概率最大的行为
		action = np.argmax(action_value)
		# 将神经网络选择出的数据记录到表中
		if not hasattr(self , "q"):
			self.q = []
			self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_value)

		self.q.append(self.running_q)

		# 判断是否随机选择动作 
		if np.random.uniform() > self.epsilon:
			action = np.random.randint(0,self.n_action)

		return action

		pass

	# 让神经网络学习
	def learn(self):

		# 达到指定训练次数，更新参数
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)

		# 取出存储的数据，如果在指定数量内，则从最大值开始随机选择，如果满足最大值则随机选择 
		# np.random.choice 第一个参数随机数的最大值，第二个参数 需要的随机数数量
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size = self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

		batch_memory = self.memory[sample_index,:]

		# 运行评估神经网络 和 目标神经网络 获得运算结果
		q_next , q_eval4next = self.sess.run([self.q_next , self.q_eval] , feed_dict = {
			# 这个地方获取全部行数据，的从-self.n_features开始的所有列数据，因为数据最后几位才是环境反馈的数据
			self.s_ : batch_memory[:,-self.n_features:], # 取训练数据中的 s_
			self.s : batch_memory[: , -self.n_features:] # 取训练数据中的 s_
			} )

		# 运行 目标神经网络，将数据集放入运算结果
		q_eval = self.sess.run(self.q_eval , {self.s : batch_memory[ : , :self.n_features]}) # 取训练数据中的 s
		# 将目标神经网络的运算数据复制
		q_target = q_eval.copy()
		# 生成 batch_size 相同的index
		batch_index = np.arange(self.batch_size , dtype = np.int32)
		# 获取self.n_features位置上的参数 也就是 a 
		eval_act_index = batch_memory[: ,self.n_features].astype(int)
		# 获取收集数据中的 r 环境反馈值
		rewarad = batch_memory[:, self.n_features +1]

		if self.double_q :
			max_act4next = np.argmax(q_eval4next , axis = 1)
			selected_q_next = q_next[batch_index , max_act4next]
		else:
			selected_q_next = np.max(q_next , axis =1)

		# 因为 q_target 是对目标神经网络运算值的复制，所以它的值也是神经网络计算出来环境状态的输出结果，
		# 所以可以取到 batch_memory 记录数据action 对应的值
		# 下面就是将对应的值目标函数运行结果的值 重新计算赋值
		q_target[batch_index , eval_act_index ] = rewarad + self.gamma * selected_q_next

		# 运算损失函数，最小化损失函数
		_,self.cost = self.sess.run([self._train_op , self.loss],
									feed_dict = { self.s : batch_memory[: , :self.n_features],
									self.q_target: q_target})
		# 将损失值记录，方便统计
		self.cost_his.append(self.cost)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

		self.learn_step_counter += 1
		pass





























