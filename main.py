import argparse

import gym
import gym_mergablerobots
import torch
from gym.wrappers import FlattenObservation, FilterObservation
import numpy as np

from TD3 import TD3


def main(args):
    env = gym.make(args.env, reward_type=args.reward_type)
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
    }

    # Initialize policy
    if args.policy == "TD3":
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = TD3.DDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = TD3.DDPG.DDPG(**kwargs)

    policy_file = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Loading existing model from: ./models/{policy_file}")
    print("---------------------------------------")
    policy.load(f"./models/{policy_file}")

    success_counter = 0
    for _ in range(1000):
        state, done, is_success = env.reset(), False, False
        while not done and not is_success:
            action = policy.select_action(np.array(state))
            state, reward, done, info = env.step(action)
            is_success = info['is_success']
            env.render()
            success_counter += int(is_success)
    print(success_counter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="UrBinPickingPlace-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1234, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--reward_type", default='place')
    parser.add_argument("--render", action="store_true")  # Render the Training
    parser.set_defaults(render=True)
    args = parser.parse_args()


    main(args)


