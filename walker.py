# Sound Source locate
# 
# @Time    : 2019-10-11 18:55
# @Author  : xyzhao
# @File    : walker.py
# @Description: define walker as an rl agent with actor-critic framework

import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle
import math
import numpy as np
import os
from bin_classfic import BinSupervisor
from astar import Node, Astar

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)

"""
    init from supervised learning model

    Actor takes states (none, n_features) as input 
    output distribution on (none, n_actions)
    
    train with feed dict:
        states,
        chosen action,
        td_error, (from critic)
"""


class Actor:
    def __init__(self, n_features, n_actions, lr):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='state')  # [1, n_F]
        self.a = tf.placeholder(tf.int32, None, name='action')  # None
        self.td_error = tf.placeholder(tf.float32, None, name='td-error')  # None

        # restore from supervised learning model
        with tf.variable_scope('Supervised'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=int(math.sqrt(self.n_actions * self.n_features)),
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        # define new loss function for actor
        with tf.variable_scope('actor_loss'):
            log_prob = tf.log(self.acts_prob[0, self.a] + 0.0000001)  # self.acts_prob[0, self.a]
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        # fixme, when load all variables in, we need reset optimizer
        with tf.variable_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(-self.exp_v)

            self.reset_optimizer = tf.variables_initializer(optimizer.variables())

        self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load_trained_model(self, model_path):
        # fixme, when load models, variables are transmit: layers, adam (not placeholder and op)
        self.saver.restore(self.sess, model_path)
        # load l1, acts_prob and adam vars
        # fixme, after load, init adam
        self.sess.run(self.reset_optimizer)

    # invalid indicates action index
    def output_action(self, s, invalid_actions):
        acts = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        # fixme, mask invalid actions based on invalid actions
        p = acts.ravel()
        p = np.array(p)

        for i in range(self.n_actions):
            if i in invalid_actions:
                p[i] = 0

        # choose invalid action with possible 1
        if p.sum() == 0:
            print("determine invalid action")
            act = np.random.choice(np.arange(acts.shape[1]))
        else:
            p /= p.sum()
            act = np.random.choice(np.arange(acts.shape[1]), p=p)
            # act = np.argmax(p)

        return act, p

    def learn(self, s, a, td):
        # fixme, may modify s
        # s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict=feed_dict)

"""
    Critic takes states (none, n_features) as input
    output td_error (float value)
    
    train with feed dict:
        states,
        new_states,
        reward
    Loss : minimize square td_error
"""


class Critic:
    def __init__(self, n_features, n_actions, lr, gamma):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.v_ = tf.placeholder(tf.float32, [None, 1], name='v_next')  # [1,1]
        self.r = tf.placeholder(tf.float32, None, name='reward')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=int(math.sqrt(1 * self.n_features)),
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='v'
            )

        with tf.variable_scope('td_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('critic_optimizer'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # fixme, global will init actor vars, partly init
        # fixme, need init: layer, optimizer (placeholder and op init is unnecessary)
        # self.sess.run(tf.global_variables_initializer())
        uninitialized_vars = [var for var in tf.global_variables() if 'critic' in var.name or 'Critic' in var.name]

        initialize_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(initialize_op)

    def learn(self, s, r, s_):
        # fixme, need modify s, s_
        # s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error


# actor = Actor(366, 8, lr=0.0001)
# actor.load_trained_model("save/4x3x4_src_0_1.6_0/save800.ckpt")
# critic = Critic(366, 8, lr=0.0001, gamma=0.95)
# action = actor.output_action([4])
# # step new state
# td = critic.learn(c, 34.2, c_)
# actor.learn(c, action, td)


"""
    walker, 
        define current pos (init with y=1)
        observation read from env.plk
        obtain reward from Game (walker doesn't know source sound pos)
"""


class Walker:
    def __init__(self, n_features, n_actions):
        self.n_features = n_features
        self.n_actions = n_actions

        # env, simulate observations
        env = open('env_hole.pkl', 'rb')
        self.observe_env = pickle.load(env)
        env.close()

        env = open('env_hole_vol.pkl', 'rb')
        self.observe_vol = pickle.load(env)
        env.close()

        # current position
        self.pos_x = None
        self.pos_y = 1.0
        self.pos_z = None

        # 8 action dim
        self.action_labels = ['0', '45', '90', '135', '180', '225', '270', '315']

        self.actor = Actor(self.n_features, self.n_actions, lr=0.004)
        self.actor.load_trained_model("save/multiple/hole/save100.ckpt")

        # fixme, first define critic before load : will report bug for not found in checkpoint
        self.critic = Critic(self.n_features, self.n_actions, lr=0.003, gamma=0.95)

        # fixme, use trained model to predict
        self.bin_graph = tf.Graph()
        with self.bin_graph.as_default():
            self.bin_classfic = BinSupervisor(366, 2)

        # fixme, avoid obstacles, env defined in A star
        self.astar = Astar()

    def reset_walker_pos(self, x, y, z):
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z

    # '-2.0_1_2.0':[gcc_vector, label]
    def observe_gcc_vector(self, x, y, z):
        # pick as key
        key = str(float(x)) + "_" + str(y) + "_" + str(float(z))
        return self.observe_env[key][0]

    # '-2.0_1_2.0':vol
    def observe_volume(self, x, y, z):
        # return is a 4-dim vector for each mic
        key = str(float(x)) + "_" + str(y) + "_" + str(float(z))
        return self.observe_vol[key]

    def choose_action(self, s, invalid_actions):
        a, p = self.actor.output_action(s, invalid_actions)
        return a, p

    def learn(self, s, a, s_, r):
        td = self.critic.learn(s, r, s_)
        self.actor.learn(s, a, td)

    # fixme, call binary model to judge in room or not
    # use argmax to determine
    def sound_in_room(self, x):
        with self.bin_graph.as_default():
            acts = self.bin_classfic.is_in_room(x)
            if np.argmax(acts) == 0:
                return True
            else:
                return False

    def find_shortest_path(self, sx, sz, dx, dz):
        return self.astar.find_path(sx, sz, dx, dz)


if __name__ == '__main__':
    # add new dim when feed s or s' if only single batch
    # c = np.array([float(i / 1000) for i in range(366)])
    # print(c)
    # c = c[np.newaxis, :]
    #
    # c_ = c
    walker = Walker(366, 8)
    x = walker.observe_gcc_vector(-2.0, 1, -3.0)
    print(walker.sound_in_room(x))
    x = np.array(x)[np.newaxis, :]
    print(walker.choose_action(x, []))
    # walker.observe_volume(2.0, 1, 2.0)
    # a = walker.choose_action(c, [4])
    # walker.learn(c, a, c_, 34)
