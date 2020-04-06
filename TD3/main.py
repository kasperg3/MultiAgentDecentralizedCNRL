import argparse
import os
import time
from collections import deque

import utils
import gym
import numpy as np
import torch
import TD3
import DDPG
import OurDDPG

import gym_mergablerobots
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
from gym.wrappers import FlattenObservation, FilterObservation


def eval_policy(policy, reward_type, env_name, seed, eval_episodes=15):
	eval_env = gym.make(env_name, reward_type=str(reward_type))
	eval_env = FlattenObservation(FilterObservation(eval_env, ['observation', 'desired_goal']))

	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done, is_success, is_failed = eval_env.reset(), False, False, False
		while not done and not is_success and not is_failed:
			action = policy.select_action(np.array(state))
			state, reward, done, info = eval_env.step(action)
			is_success = info['is_success']
			is_failed = info['is_failed']
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="UrBinPickingLift-v0")  	# OpenAI gym environment name
	parser.add_argument("--reward", default="lift")      			# reward type
	parser.add_argument("--seed", default=1234, type=int)           # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=7500000, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--render", action="store_true")            # Render the Training
	parser.set_defaults(load_model='')  							# Set to "default" if you want to load default model
	parser.set_defaults(render=True)
	parser.set_defaults(save_model=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env, reward_type=args.reward)
	env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	print("---------------------------------------")
	print(f"Environment Details: state dim: {kwargs['state_dim']}, action dim: {kwargs['action_dim']}, max action: {kwargs['max_action']}")
	print("---------------------------------------")


	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = TD3.DDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = TD3.DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		print("---------------------------------------")
		print(f"Loading existing model from: ./models/{policy_file}")
		print("---------------------------------------")
		policy.load(f"./models/{policy_file}")
	else:
		print("---------------------------------------")
		print(f"Creating new training session")
		print("---------------------------------------")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.reward, args.env, args.seed)]
	best_eval = evaluations[0]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# reset timer and init time queue
	episode_real_time = time.time()
	episode_time_buffer = deque([], maxlen=10)

	for t in range(int(args.max_timesteps)):

		if args.render == True:
			env.render()

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
					policy.select_action(np.array(state)) +
					np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, info = env.step(action)
		done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		# If the episode is done or the agent reaches a terminal state or info['is_success']
		if done or info['is_failed'] or info['is_success']:
			episode_time_buffer.append(time.time() - episode_real_time)
			est_time_left = ((sum(episode_time_buffer)/episode_time_buffer.maxlen)/150) * (args.max_timesteps - t)
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(  f"Total T: {t+1} | "
					f"Episode Num: {episode_num+1} | "
				   	f"Episode T: {episode_timesteps} | "
				   	f"Reward: {episode_reward:.3f} | "
				   	f"Episode time: {time.time() - episode_real_time:.2f} [seconds] | "
				   	f"Estimated time left: {est_time_left/60:.2f} [minutes] | "
					f"Success:  {info['is_success']} | "
					f"Failed:  {info['is_failed']}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			episode_real_time = time.time()

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.reward, args.env, args.seed))
			np.save(f"./results/{file_name}_test", evaluations)
			np.save(f"./results/{file_name}_train", episode_reward)
			# TODO: Only save the best evaluation of the model
			if args.save_model:
				policy.save(f"./models/{file_name}")
				print(".............Saving model..............")
				print("---------------------------------------")
			episode_real_time = time.time()
