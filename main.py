import os, sys, time
import json
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

        score_history = []

        tensorboard_dir = './' + args['env'] + '_' + args['variation'] + '/train_' + datetime.now().strftime(
            '%Y-%m-%d-%H')
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
                episode_score += reward

                #Update the next state
                state = state_next

                # render episode
                if args['render']:
                    env.render()
            score_history.append(episode_score)
            print('episode ', i, 'score %.2f' % episode_score,
                  'trailing ' + '100' + ' games avg %.3f' % np.mean(score_history[-100:]))
            if i % 25 == 0:
                agent.save_checkpoint()
                saver.save(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))

    # close gym
    env.close()
    sess.close()

    return
