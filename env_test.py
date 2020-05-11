import gym
import gym_mergablerobots
import torch
import numpy as np
env = gym.make('Concept-v0')
env.reset()
render = False

#"REACH": 0,
#"ORIENT": 1,
#"LIFT": 2,
#"PLACE": 3,
#"CLOSE_GRIPPER": 4,
#"OPEN_GRIPPER": 5,
#"NOOP": 6,
#"HOME": 7,
overall_success_array = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
for test in range(1):
    test = 3
    success_array = np.array([0, 0, 0])
    reward_array = np.array([0, 0])
    for episode in range(1000):
        #if render:
        #    env.render()
        env.step([5, 5])  # 5 = open gripper
        if test == 0 or test == 2:
            env.step([6, 0])
            env.step([6, 1])
            env.step([6, 4])  # 4 = close gripper
            env.step([6, 2])
            env.step([6, 3])
            env.step([6, 3])
            #env.step([6, 7])
        if test == 1 or test == 2:
            env.step([0, 6])
            env.step([1, 6])
            env.step([4, 6])  # 4 = close gripper
            env.step([2, 6])
            env.step([3, 6])
            env.step([3, 6])
            #env.step([7, 6])
        if test == 3:
            env.step([0, 5])
            env.step([1, 5])
            env.step([4, 5])  # 4 = close gripper
            env.step([2, 5])
            env.step([3, 5])
            env.step([3, 5])
            env.step([3, 5])
            #env.step([7, 6])
            env.step([4, 0])
            env.step([4, 1])
            env.step([4, 4])  # 4 = close gripper
            env.step([4, 2])
            env.step([4, 3])
            env.step([4, 3])
            env.step([4, 3])
            #env.step([6, 7])
        episode_reward = 0
        episode_success_array = [0, 0, 0]

        agent0_reward = env.compute_agent_reward(agent='0')
        agent1_reward = env.compute_agent_reward(agent='1')
        np.append(reward_array, [agent0_reward, agent1_reward], axis=0)
        print('test: ', test, ' | episode number: ', episode, ' | agent0 reward: ', agent0_reward, ' | agent1 reward: ', agent1_reward)

        if agent0_reward == 1:
            episode_success_array[0] += 1
        if agent1_reward == 1:
            episode_success_array[1] += 1
        if agent0_reward == 1 and agent1_reward == 1:
            episode_success_array[2] += 1
        success_array += episode_success_array
        env.reset()
    overall_success_array[test] = success_array
    print('success array', success_array)
print('overall: ', overall_success_array)
env.close()
