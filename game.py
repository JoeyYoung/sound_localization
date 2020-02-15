# Sound Source locate
# 
# @Time    : 2019-10-11 18:51
# @Author  : xyzhao
# @File    : game.py
# @Description: A platform to perform reinforcement learning

import numpy as np
import collections
import math
import pickle
from walker import Walker

"""
    env computing, reward computing
    game play settings

"""


class Game:
    def __init__(self):
        self.n_features = 366
        self.n_actions = 8
        self.max_epoch = 100
        self.max_steps = 100

        # define sound source information
        self.src_pos_x = -3.0
        self.src_pos_y = 1.6
        self.src_pos_z = -3.0

        # sample as a grid map with 0.5m unit
        self.unit = 0.5
        self.room_grids = [i for i in np.arange(-3.5, 3.5 + self.unit, self.unit)]

        self.walker = Walker(self.n_features, self.n_actions)

    def detect_invalids(self, x, y, z):
        invalids = []
        if x == 3.5:
            invalids.append(self.walker.action_labels.index('90'))
            invalids.append(self.walker.action_labels.index('45'))
            invalids.append(self.walker.action_labels.index('135'))
        if x == -3.5:
            invalids.append(self.walker.action_labels.index('270'))
            invalids.append(self.walker.action_labels.index('225'))
            invalids.append(self.walker.action_labels.index('315'))
        if z == 3.5:
            invalids.append(self.walker.action_labels.index('180'))
            invalids.append(self.walker.action_labels.index('135'))
            invalids.append(self.walker.action_labels.index('225'))
        if z == -3.5:
            invalids.append(self.walker.action_labels.index('0'))
            invalids.append(self.walker.action_labels.index('315'))
            invalids.append(self.walker.action_labels.index('45'))

        obstable_x = [-1, -0.5, 0, 0.5, 1]
        obstable_z = [-1, -0.5, 0, 0.5, 1]

        if x == 1.5 and z == 1.5:
            invalids.append(self.walker.action_labels.index('315'))
        elif x == 1.5 and z == 1:
            invalids.append(self.walker.action_labels.index('315'))
            invalids.append(self.walker.action_labels.index('270'))
        elif x == 1.5 and z in np.arange(-0.5, 1, 0.5):
            invalids.append(self.walker.action_labels.index('315'))
            invalids.append(self.walker.action_labels.index('270'))
            invalids.append(self.walker.action_labels.index('225'))
        elif x == 1.5 and z == -1:
            invalids.append(self.walker.action_labels.index('225'))
            invalids.append(self.walker.action_labels.index('270'))
        elif x == 1.5 and z == -1.5:
            invalids.append(self.walker.action_labels.index('225'))
        elif x == 1 and z == -1.5:
            invalids.append(self.walker.action_labels.index('225'))
            invalids.append(self.walker.action_labels.index('180'))
        elif x in np.arange(-0.5, 1, 0.5) and z == -1.5:
            invalids.append(self.walker.action_labels.index('225'))
            invalids.append(self.walker.action_labels.index('180'))
            invalids.append(self.walker.action_labels.index('135'))
        elif x == -1 and z == -1.5:
            invalids.append(self.walker.action_labels.index('180'))
            invalids.append(self.walker.action_labels.index('135'))
        elif x == -1.5 and z == -1.5:
            invalids.append(self.walker.action_labels.index('135'))
        elif x == -1.5 and z == -1:
            invalids.append(self.walker.action_labels.index('90'))
            invalids.append(self.walker.action_labels.index('135'))
        elif x == -1.5 and z in np.arange(-0.5, 1, 0.5):
            invalids.append(self.walker.action_labels.index('90'))
            invalids.append(self.walker.action_labels.index('135'))
            invalids.append(self.walker.action_labels.index('45'))
        elif x == -1.5 and z == 1:
            invalids.append(self.walker.action_labels.index('45'))
            invalids.append(self.walker.action_labels.index('90'))
        elif x == -1.5 and z == 1.5:
            invalids.append(self.walker.action_labels.index('45'))
        elif x == -1 and z == 1.5:
            invalids.append(self.walker.action_labels.index('0'))
            invalids.append(self.walker.action_labels.index('45'))
        elif x in np.arange(-0.5, 1, 0.5) and z == 1.5:
            invalids.append(self.walker.action_labels.index('315'))
            invalids.append(self.walker.action_labels.index('0'))
            invalids.append(self.walker.action_labels.index('45'))
        elif x == 1 and z == 1.5:
            invalids.append(self.walker.action_labels.index('315'))
            invalids.append(self.walker.action_labels.index('0'))

        # todo, abstract an obstacle
        return invalids

    def play(self):
        records_step = []
        records_r = []

        for epoch in range(self.max_epoch):
            print("========== Epoch %d ======" % epoch)
            memory = collections.defaultdict(dict)
            visit = {}
            for i in self.room_grids:
                for j in self.room_grids:
                    visit[str(i) + "*" + str(j)] = 0
                    for k in self.walker.action_labels:
                        memory[str(i) + "*" + str(j)][k] = 0

            # init walker position
            # fixme, random choose
            self.walker.reset_walker_pos(3.0, 1, 3.0)
            DONE = False
            sum_reward = 0.0

            a_his = None
            for step in range(self.max_steps):
                s = self.walker.observe_gcc_vector(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                s = np.array(s)[np.newaxis, :]

                # fixme, use grids to detect
                # fixme, cut action space
                invalids = self.detect_invalids(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)

                pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)
                for i in memory[pos_key].keys():
                    if memory[pos_key][i] >= 2:
                        invalids.append(self.walker.action_labels.index(i))
                visit[pos_key] += 1

                a, p = self.walker.choose_action(s, invalids)

                # step next state
                direction = self.walker.action_labels[a]

                # fixme, for the first step, give more obs, argmax
                if step == 0:
                    fe = open('first_obs.pkl', 'rb')
                    obs = pickle.load(fe)

                    s_r = obs['right']
                    s_r = np.array(s_r)[np.newaxis, :]
                    a_r, p_r = self.walker.choose_action(s_r, [])
                    p_rr = [p_r[len(p_r) - 2], p_r[len(p_r) - 1]]
                    p_rr = np.append(p_rr, p_r[:len(p_r) - 2])

                    s_l = obs['left']
                    s_l = np.array(s_l)[np.newaxis, :]
                    a_l, p_l = self.walker.choose_action(s_l, [])
                    p_ll = [p_l[0], p_l[1]]
                    p_ll = np.append(p_l[2:], p_ll)

                    s_d = obs['down']
                    s_d = np.array(s_d)[np.newaxis, :]
                    a_d, p_d = self.walker.choose_action(s_d, [])
                    p_dd = [p_d[len(p_d) - 4], p_d[len(p_d) - 3], p_d[len(p_d) - 2], p_d[len(p_d) - 1]]
                    p_dd = np.append(p_dd, p_d[:len(p_d) - 4])

                    # fixme, define first step based on obs, do argmax
                    p_mix = [0] * self.n_actions
                    for i in range(self.n_actions):
                        if i not in invalids:
                            p_mix[i] = p[i] + p_rr[i] + p_ll[i] + p_dd[i]

                    p_mix = np.array(p_mix)
                    p_mix /= p_mix.sum()
                    a_mix = np.argmax(p_mix)

                    fe.close()

                    a = a_mix
                    a_his = a
                    p = p_mix
                    direction = self.walker.action_labels[a]

                # if epoch == 20:
                #     print(p)
                #     print(direction)

                memory[pos_key][direction] += 1

                if direction == '0':
                    self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                 self.walker.pos_z - self.unit)
                elif direction == '45':
                    self.walker.reset_walker_pos(self.walker.pos_x + self.unit, self.walker.pos_y,
                                                 self.walker.pos_z - self.unit)
                elif direction == '90':
                    self.walker.reset_walker_pos(self.walker.pos_x + self.unit, self.walker.pos_y,
                                                 self.walker.pos_z)
                elif direction == '135':
                    self.walker.reset_walker_pos(self.walker.pos_x + self.unit, self.walker.pos_y,
                                                 self.walker.pos_z + self.unit)
                elif direction == '180':
                    self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                 self.walker.pos_z + self.unit)
                elif direction == '225':
                    self.walker.reset_walker_pos(self.walker.pos_x - self.unit, self.walker.pos_y,
                                                 self.walker.pos_z + self.unit)
                elif direction == '270':
                    self.walker.reset_walker_pos(self.walker.pos_x - self.unit, self.walker.pos_y,
                                                 self.walker.pos_z)
                elif direction == '315':
                    self.walker.reset_walker_pos(self.walker.pos_x - self.unit, self.walker.pos_y,
                                                 self.walker.pos_z - self.unit)

                # fixme, don't have s_ when get source
                if self.walker.pos_x == self.src_pos_x and self.walker.pos_z == self.src_pos_z:
                    print("get source")
                    DONE = True
                    r = 5
                    s_ = np.array([0 for u in range(self.n_features)])[np.newaxis, :]

                else:
                    # fixme, rebuild reward function
                    # r = self.walker.observe_volume(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                    # r = 0
                    pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)
                    # r /= (visit[pos_key] + 1)

                    max_angle = max(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))
                    min_angle = min(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))

                    diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)

                    # r = 1 - abs((a + a_his) % self.n_actions - a_his) / (self.n_actions - 1)
                    r = 1 - diff / 180
                    r -= (visit[pos_key]) * 0.2

                    # todo, think about punishment
                    pub = self.detect_invalids(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                    if len(pub) > 0:
                        r -= 0.5

                    s_ = self.walker.observe_gcc_vector(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                    s_ = np.array(s_)[np.newaxis, :]

                sum_reward += r
                a_his = a

                self.walker.learn(s, a, s_, r)
                if DONE:
                    break

            # fixme, think about a new way to evaluate
            print(step)
            print(sum_reward / step)
            records_step.append(step)
            records_r.append(sum_reward / step)

            # overload now
            if epoch % 500 == 0 and epoch != 0:
                with open('save/rl_8x3x8_src_-3_1.6_-3/records_step', 'w') as f:
                    f.write(str(records_step))
                with open('save/rl_8x3x8_src_-3_1.6_-3/records_reward', 'w') as f:
                    f.write(str(records_r))


if __name__ == '__main__':
    game = Game()
    # game.detect_invalids(1, 1, 1)
    game.play()
