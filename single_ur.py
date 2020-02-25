import os, sys, time
import json
import pickle

import gym
from gym import wrappers
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent import Actor, Critic, Agent
from replay import Memory
from agent import Noise
import matplotlib.pyplot as plt

# function to unpack observation from gym environment
def unpackObs(obs):
    return  obs['achieved_goal'], \
            obs['desired_goal'],\
            np.concatenate((obs['observation'], \
            obs['desired_goal'])), \
            np.concatenate((obs['observation'], \
            obs['achieved_goal']))

# Main
def main(args):
    # Set path to save result
    gym_dir = './' + args['env'] + '_' + args['variation'] + '/gym'

    # Set random seed for reproducibility
    np.random.seed(int(args['seed']))
    tf.set_random_seed(int(args['seed']))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set random seed for reproducibility
    np.random.seed(int(args['seed']))
    tf.set_random_seed(int(args['seed']))

    # Load environment
    env = gym.make(args['env'])
    env.seed(int(args['seed']))

    sess = tf.Session()

    with sess:
        agent = Agent(args, sess, env=env)
        np.random.seed(0)

        tensorboard_dir = './' + args['env'] + '_' + args['variation'] + '/train_' + datetime.now().strftime(
            '%Y-%m-%d-%H') + 'seed_' + str(args['seed'])
        model_dir = './' + args['env'] + '_' + args['variation'] + '/model'
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)  # session?

        # initialize variables, create writer and saver
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

        try:
            saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))
            print('Restore from previous training session')
        except:
            print('Start new training session')
        epoch_history = []
        for k in range(args['epochs']):
            score_history = []
            for i in range(args['episodes']):
                achieved_goal, desired_goal, state, state_prime = unpackObs(env.reset())
                done = False
                episode_score = 0
                for j in range(int(args['episode_length'])):
                    act = agent.choose_action(state=state)
                    new_obs, reward, done, info = env.step(act[0])
                    achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(new_obs)

                # Store data in replay buffer
                agent.remember(state, state_next, act[0], reward, done)
                agent.rememberHER(state_prime, state_prime_next, achieved_goal, info, act[0], env)

                    agent.learn()

                    #Update the next state and add reward to episode_score
                    episode_score += reward
                    state = state_next

                    # render episode
                    if args['render']:
                        env.render()

                #Save the episode scores
                score_history.append(episode_score)
                print('epoch:' + str(k) + ' | episode: ', str(i), ' | score %.2f' % episode_score, )
            # Take the mean of the scores and normalize it in the epoch and save it
            epoch_history.append(np.divide(score_history, int(args['episode_length'])))

            #Save model each epoch
            agent.save_checkpoint()
            saver.save(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))

    #plot the mean of the epochs
    epoch_plot, = plt.plot(range(args['epochs']), np.mean(epoch_history, axis=1), color='blue')
    std_deviation = np.std(epoch_history, axis=1)
    plt.fill_between(range(args['epochs']),
                     np.subtract(np.mean(epoch_history, axis=1), std_deviation),
                     np.add(np.mean(epoch_history, axis=1), std_deviation), facecolor='blue', alpha=0.1)
    plt.ylabel('Reward')
    plt.xlabel('Epoch')
    plt.legend([args['variation']])
    plt.show()
    plt.pause(0.05)

    pickle.dump(epoch_plot, open("./plot_data/plot_data_" + args['variation'], "wb"))

    input("Press to exit")

    # close gym
    env.close()
    sess.close()

import gym_mergablerobots.envs.mergablerobots.ur_reach

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--actor-lr', help='actor learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic learning rate', default=0.001)
    parser.add_argument('--batch-size', help='batch size', default=64)
    parser.add_argument('--gamma', help='discount factor reward', default=0.99)
    parser.add_argument('--tau', help='target update tau', default=0.001)
    parser.add_argument('--memory-size', help='size of the replay memory', default=1000000)
    parser.add_argument('--hidden-sizes', help='number of nodes in hidden layer', default=(400, 300))
    parser.add_argument('--episode-length', help='max length of 1 episode', default=150)
    parser.add_argument('--episodes', help='episodes to train', default=20)
    parser.add_argument('--epochs', help='number of epochs', default=100)

    # others and defaults
    parser.add_argument('--seed', help='random seed', default=1235)
    parser.add_argument('--render', help='render the gym env', action='store_true')
    parser.add_argument('--test', help='test mode does not do exploration', action='store_true')
    parser.add_argument('--variation', help='model variation name', default='DDPG_HER')
    #parser.set_defaults(env='FetchReach-v1')
    #parser.set_defaults(env='mergablerobots-v0')
    parser.set_defaults(env='UrReach-v0')
    #parser.set_defaults(env='FetchPickAndPlace-v1')
    parser.set_defaults(render=False)
    parser.set_defaults(test=False)

    # parse arguments
    args = vars(parser.parse_args())

    # run main
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
