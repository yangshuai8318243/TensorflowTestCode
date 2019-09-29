# -*- coding: utf-8 -*-
# @Author: [ShuaiYang]
# @Date:   2019-09-27 16:54:54
# @Last Modified by:   [ShuaiYang]
# @Last Modified time: 2019-09-29 16:54:50
from BoxEnv import Box 
from Rl_brain import QLearingTable


def update():
	for x in range(110):
		#每次训练初始化位置
		observation = env.reset()

		while True:
			#更新一下状态
			env.render()

			#通过q表选择下一步的动作
			action = RL.choose_action(str(observation))

			# 通过q表获取的值，在环境中进行一点，并且得到反馈
			observation_, reward, done = env.step(action)

            # 让Q表通过反馈进行学习
			RL.learn(str(observation), action, reward, str(observation_))


            # 将状态改变
			observation = observation_
			
			#如果无法进行则停止循环
			if done:
				print(x,"stop game")
				break
			pass

		pass
	print('game over')
	env.destroy()

	pass

if __name__ == "__main__":
	#初始化对象
    env = Box()
    RL = QLearingTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()