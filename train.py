# Sound Source locate
#
# @Time    : 2019-10-9 17:21
# @Author  : xyzhao
# @File    : train.py
# @Description: perform supervised learning

import tensorflow as tf
import numpy as np
import math
import os
import random

# x = np.array([[-29, 22, 332, -4, 225, 63],
#               [-9, 222, 32, -432, 5, 33]])
# y = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 0, 0]])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)


class Supervisor:
    """
        init parameters, layers
    """

    def __init__(self, n_features, n_actions, lr=0.0001):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.max_epochs = 50

        # fixme, use rl as dataset
        self.train_data_size = 199
        self.test_data_size = 0  # 400
        self.batch_size = 1

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='gcc')
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], name='label')

        with tf.variable_scope('Supervised'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=int(math.sqrt(self.n_features * self.n_actions)),
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

        with tf.variable_scope('spv_loss'):
            acts_clip = tf.clip_by_value(self.acts_prob, 1e-10, 0.9999999)
            self.loss = -tf.reduce_mean(
                tf.reduce_sum(self.a * tf.log(acts_clip) + (1 - self.a) * tf.log(1 - acts_clip), axis=1))
        # self.loss = tf.reduce_mean(
        #     -tf.reduce_sum(self.a * tf.log(self.acts_prob + 0.0000001), reduction_indices=[1]))

        with tf.variable_scope('adam_optimizer'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

        # fixme, save all variables above (vars: layers, adam. not op and placeholder)
        self.saver = tf.train.Saver()

    """
        read from file into self. x, self. y; 
        vector_train=false, read vector_test data
        note: tail line need a blank line
    """

    def load_data(self, dir, file, train=True):
        if train is True:
            data_size = self.train_data_size
        else:
            data_size = self.test_data_size

        self.x = [None] * data_size
        self.y = [None] * data_size

        with open(os.path.join(dir, file), 'r') as f:
            lines = f.readlines()
            for i in range(data_size):
                # for i-th line
                self.x[i] = []
                self.y[i] = []
                line = lines[i][2:len(lines[i]) - 3]
                contents = line.split(',')

                # scan each content
                for p in range(self.n_features):
                    if p == self.n_features - 1:
                        content = contents[p][:len(contents[p]) - 1]
                    else:
                        content = contents[p]
                    self.x[i].append(float(content))
                for p in range(self.n_features, len(contents)):
                    if p == self.n_features:
                        content = contents[p][2:]
                    else:
                        content = contents[p]
                    self.y[i].append(int(content))

    """
        load vector_train data "vector_train", can batch learning
        save save every 100 epoch
    """

    def train_step(self):
        # self.load_data("./gccdata/multiple/hole_eight", "vector_train", train=True)
        # self.load_data("./audio/gcc", "rscback_aaaa", train=True)
        self.load_data("./gcc/cyc4", "cyc4", train=True)
        records_acu = []
        records_los = []

        if self.train_data_size % self.batch_size == 0:
            round = int(self.train_data_size / self.batch_size)
        else:
            round = int(self.train_data_size / self.batch_size) + 1

        indexs = [i for i in range(self.train_data_size)]

        """
            begin time epoch
        """
        for epoch in range(self.max_epochs + 1):
            print("========== Epoch %d ======" % epoch)
            correct = 0.0
            sum_loss = 0

            random.shuffle(indexs)

            for i in range(round):

                # if not batch, need add lim
                # s = x[np.newaxis, :]
                # a = y[np.newaxis, :]
                if i == (round - 1):
                    temp = [k for k in indexs[i * self.batch_size:]]
                    x = [self.x[p] for p in temp]
                    y = [self.y[p] for p in temp]
                else:
                    temp = [k for k in indexs[i * self.batch_size:(i + 1) * self.batch_size]]
                    x = [self.x[p] for p in temp]
                    y = [self.y[p] for p in temp]

                # may bug ,x y only single
                feed_dict = {self.s: np.array(x), self.a: np.array(y)}
                _, loss, acts = self.sess.run([self.train_op, self.loss, self.acts_prob], feed_dict=feed_dict)
                sum_loss += loss

                for p in range(len(y)):
                    if np.argmax(acts[p]) == np.argmax(y[p]):
                        correct += 1
                # run train_op will auto run loss and acts,
                # but if we want to get loss and acts, need additionally run them

            print("loss: " + str(sum_loss / self.train_data_size))
            print("accuracy: " + str(correct / self.train_data_size))

            records_acu.append(correct / self.train_data_size)
            records_los.append(sum_loss / self.train_data_size)
            if epoch % 10 == 0 and epoch != 0:
                self.saver.save(self.sess, "save/cyc4/save%d.ckpt" % epoch)

        # with open('save/multiple/eight/records_acu', 'w') as f:
        #     f.write(str(records_acu))
        # with open('save/multiple/eight/records_los', 'w') as f:
        #     f.write(str(records_los))

    """
        load vector_test data "vector_test", load trained save
    """

    def predict_step(self):
        print("Start predict on vector_test data ....")

        # self.load_data("./gccdata/multiple/hole_eight", "vector_test", train=False)
        self.load_data("./gcc/cyc4/", "cyc4_test", train=False)

        # self.saver.restore(self.sess, "save/multiple/hole/save10.ckpt")
        # self.saver.restore(self.sess, "save/8x3x8_9pos/save100.ckpt")
        self.saver.restore(self.sess, "./save/cyc4/save50.ckpt")

        correct = 0.0
        for i in range(self.test_data_size):
            print("===================")
            x = np.array(self.x[i])
            x = x[np.newaxis, :]
            y = np.array(self.y[i])
            y = y[np.newaxis, :]

            feed_dict = {self.s: x, self.a: y}
            acts = self.sess.run(self.acts_prob, feed_dict=feed_dict)

            print(acts)
            print("output: " + str(np.argmax(acts[0]) * 45))
            print("label: " + str(np.argmax(y[0]) * 45))
            if np.argmax(acts[0]) == np.argmax(y[0]):
                correct += 1

        print("accuracy: " + str(correct / self.test_data_size))

    # def train_step_single(self):
    #     self.load_data("./gccdata/4x3x4_src_0_1.6_0", "vector_train", vector_train=True)
    #
    #     """
    #         begin time epoch
    #     """
    #     for epoch in range(self.max_epochs + 1):
    #         print("========== Epoch %d ======" % epoch)
    #         correct = 0.0
    #         sum_loss = 0
    #
    #         for i in range(self.train_data_size):
    #
    #             # if not batch, need add lim
    #             # s = x[np.newaxis, :]
    #             # a = y[np.newaxis, :]
    #             x = np.array(self.x[i])
    #             x = x[np.newaxis, :]
    #             y = np.array(self.y[i])
    #             y = y[np.newaxis, :]
    #
    #             # may bug ,x y only single
    #             feed_dict = {self.s: x, self.a: y}
    #             _, loss, acts = self.sess.run([self.train_op, self.loss, self.acts_prob], feed_dict=feed_dict)
    #
    #             if np.argmax(acts[0]) == np.argmax(y[0]):
    #                 correct += 1
    #             sum_loss += loss
    #             # run train_op will auto run loss and acts,
    #             # but if we want to get loss and acts, need additionally run them
    #         print("loss: " + str(sum_loss / self.train_data_size))
    #         print("accuracy: " + str(correct / self.train_data_size))
    #         if epoch % 100 == 0 and epoch != 0:
    #             self.saver.save(self.sess, "save/save%d.ckpt" % epoch)


if __name__ == '__main__':
    # fixme, change features dim here,
    sup = Supervisor(366, 8)
    sup.train_step()
    # sup.predict_step()
