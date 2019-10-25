# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-27 16:54:54
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-10-09 17:25:17
from BoxEnv import Box 
from Rl_brain import QLearingTable
from Rl_brain import SarsaTable
from Rl_brain import SarsaLambdaTable


def update():
	for x in range(110):
		#每次训练初始化位置
		observation = env.reset()
		index = 1;
		while True:
			#更新一下状态
			env.render()

			#通过q表选择当前的动作
			action = RL.choose_action(str(observation))

			# 通过q表获取的值，在环境中进行一点，并且得到反馈
			observation_, reward, done,tag = env.step(action)

            # 让Q表通过反馈进行学习
			RL.learn(str(observation), action, reward, str(observation_))


            # 将状态改变
			observation = observation_
			
			#如果无法进行则停止循环
			if done:
				print(x,"stop game",index,tag)
				break
			pass
			index = index +1
		pass
	print('game over')
	env.destroy()

	pass

def sarsaUpadate():
	for x in range(110):
		#每次训练初始化位置
		observation = env.reset()
		index = 1;

		action = RL.choose_action(str(observation))

		while True:
			#更新一下状态
			env.render()


			# 通过q表获取的值，在环境中进行一点，并且得到反馈
			observation_, reward, done,tag = env.step(action)

			#通过q表选择下一次的动作
			action_ = RL.choose_action(str(observation_))

            # 让Q表通过反馈进行学习
			RL.learn(str(observation), action, reward, str(observation_),action_)


            # 将状态改变
			observation = observation_
			
			action = action_
			
			#如果无法进行则停止循环
			if done:
				print(x,"stop game",index,tag)
				break
			pass
			index = index +1
		pass
	print('game over')
	env.destroy()
	pass

if __name__ == "__main__":
	#初始化对象
    env = Box()
    # RL = QLearingTable(actions=list(range(env.n_actions)))
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, sarsaUpadate)
    env.mainloop()