import gym
import gym_mergablerobots
import torch
import numpy as np
env = gym.make('Concept-v0')
env.reset()
render = False
# while True:
#     env.render()
#     env.step([5, 5])  # 5 = open gripper
#     env.step([0, 6])
#     env.step([1, 6])
#     env.step([4, 6])  # 4 = close gripper
#     env.step([2, 6])
#     env.step([3, 6])
#     env.step([3, 6])
#     env.step([6, 0])
#     env.step([6, 1])
#     env.step([6, 4])  # close
#     env.step([6, 2])
#     env.step([6, 3])
#     env.step([6, 3])
#     env.reset()
# env.close()

success_array = np.array([0, 0, 0])
reward_array = np.array([0, 0])
for episode in range(1000):
    if render:
        env.render()

    env.step([5, 5])  # 5 = open gripper
    env.step([0, 6])
    env.step([1, 6])
    env.step([4, 6])  # 4 = close gripper
    env.step([2, 6])
    env.step([3, 6])
    env.step([3, 6])
    env.step([6, 0])
    env.step([6, 1])
    env.step([6, 4])  # 4 = close gripper
    env.step([6, 2])
    env.step([6, 3])
    env.step([6, 3])


    episode_reward = 0
    episode_success_array = [0, 0, 0]

    agent0_reward = env.compute_agent_reward(agent='0')
    agent1_reward = env.compute_agent_reward(agent='1')
    np.append(reward_array, [agent0_reward, agent1_reward], axis=0)
    print('episode number: ', episode, ' | agent0 reward: ', agent0_reward, ' | agent1 reward: ', agent1_reward)

    if agent0_reward == 1:
        episode_success_array[0] += 1
    if agent1_reward == 1:
        episode_success_array[1] += 1
    if agent0_reward == 1 and agent1_reward == 1:
        episode_success_array[2] += 1
    success_array += episode_success_array
    print('success array', success_array)
    env.reset()
env.close()
    # Evaluate episode

#np.save(f"./results/{file_name}_test", evaluations)
#np.save(f"./results/{file_name}_train", train_history)
#np.save(f"./results/{file_name}_train_success", success_history)
#print(f"success since last evaluation: {eval_success:.2f} best score: {best_eval_success}")
#if eval_success >= best_eval_success:
#    best_eval_success = eval_success
#if args.save_model:
#    policy.save(f"./models/{file_name}")
#    print(".............Saving model..............")
#    print("---------------------------------------")
