"""
    RL online training Part
"""

import tensorflow as tf
import math
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)


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

        # when load all variables in, we need reset optimizer
        with tf.variable_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(-self.exp_v)

            self.reset_optimizer = tf.variables_initializer(optimizer.variables())

        self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load_trained_model(self, model_path):
        # when load models, variables are transmit: layers, adam (not placeholder and op)
        self.saver.restore(self.sess, model_path)
        # load l1, acts_prob and adam vars
        self.sess.run(self.reset_optimizer)

    # invalid indicates action index
    def output_action(self, s, invalid_actions):
        acts = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        # mask invalid actions based on invalid actions
        p = acts.ravel()
        p = np.array(p)

        for i in range(self.n_actions):
            if i in invalid_actions:
                p[i] = 0

        # choose invalid action with possible 1
        if p.sum() == 0:
            print("determine invalid action")
            act = np.random.choice(np.arange(acts.shape[1]))
            exit(1)
        else:
            p /= p.sum()
            # act = np.random.choice(np.arange(acts.shape[1]), p=p)
            act = np.argmax(p)

        return act, p

    def learn(self, s, a, td):
        # may modify s
        # s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict=feed_dict)


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

        # global will init actor vars, partly init
        # need init: layer, optimizer (placeholder and op init is unnecessary)
        # self.sess.run(tf.global_variables_initializer())
        uninitialized_vars = [var for var in tf.global_variables() if 'critic' in var.name or 'Critic' in var.name]

        initialize_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(initialize_op)

    def learn(self, s, r, s_):
        # need modify s, s_
        # s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error
