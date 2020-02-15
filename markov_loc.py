# Sound Source locate
# 
# @Time    : 2019-12-04 13:55
# @Author  : xyzhao
# @File    : markov_loc.py
# @Description: Using RL to optimize markov localization

import random
import numpy as np
import collections
import math
import pickle
from walker import Walker


class Game:
    def __init__(self):
        self.n_features = 366
        self.n_actions = 8
        self.max_epoch = 1000
        self.max_steps = 40

        # define sound source information
        self.src_pos_x = -2.0
        self.src_pos_y = 1.6
        self.src_pos_z = -4.0

        # sample as a grid map with 0.5m unit
        # change step length to 1m
        self.unit = 1.0
        self.room_grids_x = [i for i in np.arange(-3.0, 3.0 + self.unit, self.unit)]
        self.room_grids_z = [i for i in np.arange(-4.0, 4.0 + self.unit, self.unit)]

        # define wall and obstacles
        self.wall_axis_z = {-4: [i for i in np.arange(-5.0, 6.0, 1.0)],
                            4: [i for i in np.arange(-5.0, 6.0, 1.0)],
                            0: [i for i in np.arange(-5.0, 6.0, 1.0) if i != 0]}
        self.wall_axis_x = {5: [i for i in np.arange(-4.0, 5.0, 1.0)],
                            1: [i for i in np.arange(-4.0, 5.0, 1.0) if i != -2 and i != 2],
                            -1: [i for i in np.arange(-4.0, 5.0, 1.0) if i != -2 and i != 2],
                            -5: [i for i in np.arange(-4.0, 5.0, 1.0)]}

        # define checkpoints: room gates, hall center
        self.room_gates = [[-2.0, 1, -1.0], [2.0, 1, -1.0], [-2.0, 1, 1.0], [2.0, 1, 1.0]]
        self.hall_center = [[0, 0, 0]]

        # define room zone
        self.room1_x = [i for i in np.arange(-3.5, 0, 0.5)]
        self.room1_z = [i for i in np.arange(-4.5, -1, 0.5)]

        self.room2_x = [i for i in np.arange(0.5, 4.0, 0.5)]
        self.room2_z = [i for i in np.arange(-4.5, -1, 0.5)]

        self.room3_x = [i for i in np.arange(-3.5, 0, 0.5)]
        self.room3_z = [i for i in np.arange(1.5, 5.0, 0.5)]

        self.room4_x = [i for i in np.arange(0.5, 4.0, 0.5)]
        self.room4_z = [i for i in np.arange(1.5, 5.0, 0.5)]

        self.hall_x = [i for i in np.arange(-3.5, 4.0, 0.5)]
        self.hall_z = [i for i in np.arange(-0.5, 1.0, 0.5)]

        self.walker = Walker(self.n_features, self.n_actions)

        self.BayeProb = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
        self.Pzx = {1: 1, 2: 1, 3: 1, 4: 1}

    # invalid direction, False room and defined obstacles
    def detect_invalids(self, x, y, z, room):
        invalids = []
        directions = [[x, y, z - self.unit], [x + self.unit, y, z - self.unit],
                      [x + self.unit, y, z], [x + self.unit, y, z + self.unit],
                      [x, y, z + self.unit], [x - self.unit, y, z + self.unit],
                      [x - self.unit, y, z], [x - self.unit, y, z - self.unit]]

        for direction in directions:
            # along x axis, fix z, change x
            if self.wall_axis_x.get(direction[2]) is not None:
                if direction[0] in self.wall_axis_x[direction[2]]:
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))

            # along z axis, fix x, change z
            if self.wall_axis_z.get(direction[0]) is not None:
                if direction[2] in self.wall_axis_z[direction[0]]:
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))

        if room[4] is False:
            for direction in directions:
                if (direction[0] in self.room4_x and direction[2] in self.room4_z) or (
                        direction[0] == self.room_gates[3][0] and direction[2] == self.room_gates[3][2]):
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))
        if room[3] is False:
            for direction in directions:
                if (direction[0] in self.room3_x and direction[2] in self.room3_z) or (
                        direction[0] == self.room_gates[2][0] and direction[2] == self.room_gates[2][2]):
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))
        if room[2] is False:
            for direction in directions:
                if (direction[0] in self.room2_x and direction[2] in self.room2_z) or (
                        direction[0] == self.room_gates[1][0] and direction[2] == self.room_gates[1][2]):
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))
        if room[1] is False:
            for direction in directions:
                if (direction[0] in self.room1_x and direction[2] in self.room1_z) or (
                        direction[0] == self.room_gates[0][0] and direction[2] == self.room_gates[0][2]):
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))

        if room[0] is False:
            for direction in directions:
                if direction[0] in self.hall_x and direction[2] in self.hall_z:
                    invalids.append(self.walker.action_labels.index(str(directions.index(direction) * 45)))

        return invalids

    # return 1, 2, 3, 4 room, 0-hall
    def detect_which_room(self):
        if self.walker.pos_x in self.room1_x and self.walker.pos_z in self.room1_z:
            return 1
        elif self.walker.pos_x in self.room2_x and self.walker.pos_z in self.room2_z:
            return 2
        elif self.walker.pos_x in self.room3_x and self.walker.pos_z in self.room3_z:
            return 3
        elif self.walker.pos_x in self.room4_x and self.walker.pos_z in self.room4_z:
            return 4
        elif self.walker.pos_x in self.hall_x and self.walker.pos_z in self.hall_z:
            return 0
        else:
            return -1

    """
        based on GUIDE path to learn actions:
        - learn: from inner room guide to gate; avoid obstacles
        - not learn: gate into inner room

        - reward: diff in angle
    """

    def learn_guide_actions(self, path, visit):
        a_his = None

        for pos in path:
            if path.index(pos) == len(path) - 2:
                break
            s = self.walker.observe_gcc_vector(pos[0], self.walker.pos_y, pos[1])
            s = np.array(s)[np.newaxis, :]

            pos_key = str(pos[0]) + "*" + str(pos[1])
            visit[pos_key] += 1

            pos_next = path[path.index(pos) + 1]
            s_ = self.walker.observe_gcc_vector(pos_next[1], self.walker.pos_y, pos_next[1])
            s_ = np.array(s_)[np.newaxis, :]

            # get action
            if pos_next[0] - pos[0] == 0 and pos_next[1] - pos[1] == -self.unit:
                a = 0
            elif pos_next[0] - pos[0] == self.unit and pos_next[1] - pos[1] == -self.unit:
                a = 1
            elif pos_next[0] - pos[0] == self.unit and pos_next[1] - pos[1] == 0:
                a = 2
            elif pos_next[0] - pos[0] == self.unit and pos_next[1] - pos[1] == self.unit:
                a = 3
            elif pos_next[0] - pos[0] == 0 and pos_next[1] - pos[1] == self.unit:
                a = 4
            elif pos_next[0] - pos[0] == -self.unit and pos_next[1] - pos[1] == self.unit:
                a = 5
            elif pos_next[0] - pos[0] == -self.unit and pos_next[1] - pos[1] == 0:
                a = 6
            elif pos_next[0] - pos[0] == -self.unit and pos_next[1] - pos[1] == -self.unit:
                a = 7
            else:
                print("Wrong action get from GUIDE path... ")
                a = None

            if a_his is None:
                a_his = a

            # get diff reward
            max_angle = max(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))
            min_angle = min(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))

            diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)

            r = 1 - diff / 180

            pos_key = str(pos_next[0]) + "*" + str(pos_next[1])
            r -= (visit[pos_key]) * 0.2

            self.walker.learn(s, a, s_, r)
            a_his = a

    """
        Try use different way to do
    """

    def calculate_cov(self, z_real, z_exp):
        length = len(z_real)

        y = np.array(z_real)
        x = np.array(z_exp)

        x_avg = np.average(x)
        y_avg = np.average(y)

        xy = [(x[i] - x_avg) * (y[i] - y_avg) for i in range(length)]
        cov_xy = np.sum(xy)

        pow_x = [pow(float(x[i] - x_avg), 2.0) for i in range(length)]
        theta_x = math.sqrt(np.sum(pow_x))

        pow_y = [pow(float(y[i] - y_avg), 2.0) for i in range(length)]
        theta_y = math.sqrt(np.sum(pow_y))

        r = cov_xy / (theta_x * theta_y)  # [-1, 1]
        pear = r + 1  # [0, 2]

        # the larger pear means more similar
        return pear

    # when observe 1-dim vector s, update its confidence to all rooms
    def update_bayesia(self):
        key = str(float(self.walker.pos_x)) + "_" + str(self.walker.pos_y) + "_" + str(
            float(self.walker.pos_z))

        z_real = self.walker.observe_volume(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
        for room in range(1, 5):
            simu = open('simu_r%d_vol.pkl' % room, 'rb')
            exp = pickle.load(simu)
            simu.close()

            # if meet (2,3) center in room, just use previous pzx;
            if exp.get(key) is not None:
                z_exp = exp[key]

                # cov = self.calculate_cov(z_real, z_exp)

                # 7 is a constant value, need to measure before doing experiment
                diff_vol = 7 - np.abs(np.average(z_real) - np.average(z_exp))
                if self.walker.pos_x == 2 and self.walker.pos_z == -1 and room == 1:
                    diff_vol += 5

                # larger, more similar
                self.Pzx[room] = diff_vol

        # need one learning rate
        baye_sum = 0.
        for room in range(1, 5):
            self.BayeProb[room] *= self.Pzx[room]
            baye_sum += self.BayeProb[room]

        # rescale baye prob
        for room in range(1, 5):
            self.BayeProb[room] /= baye_sum

    def play(self):
        for epoch in range(self.max_epoch):
            print("========== Epoch %d ======" % epoch)

            # init historical track
            memory = collections.defaultdict(dict)
            visit = {}
            for i in self.room_grids_x:
                for j in self.room_grids_z:
                    visit[str(i) + "*" + str(j)] = 0
                    for k in self.walker.action_labels:
                        memory[str(i) + "*" + str(j)][k] = 0

            """
                Init part
            """
            self.walker.reset_walker_pos(2.0, 1, 4.0)
            DONE = False
            sum_reward = 0.0
            a_his = None
            ROOM = [None] * 5

            self.BayeProb = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
            self.Pzx = {1: 1, 2: 1, 3: 1, 4: 1}

            """
                get first observation and update 
            """

            s = self.walker.observe_gcc_vector(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
            self.update_bayesia()
            print(self.Pzx)
            print(self.BayeProb)
            s = np.array(s)[np.newaxis, :]

            for step in range(self.max_steps):
                # Note: always has a s (init or s = s_)
                print("************** step %d" % step)

                print("x: " + str(self.walker.pos_x))
                print("z: " + str(self.walker.pos_z))

                # todo, if use A* to guide in some situations, update Bayesia, update s, continue; learn if needed

                invalids = self.detect_invalids(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z, ROOM)

                # todo, append invalids
                pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)
                for i in memory[pos_key].keys():
                    if self.detect_which_room() == 0:
                        threshold = 10
                    else:
                        threshold = 2
                    if memory[pos_key][i] >= threshold:
                        invalids.append(self.walker.action_labels.index(i))

                a, p = self.walker.choose_action(s, invalids)

                if a_his is None:
                    a_his = a

                direction = self.walker.action_labels[a]
                memory[pos_key][direction] += 1
                visit[pos_key] += 1

                """
                    Apply movement
                """

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

                print("apply movement: " + direction)

                """
                    Receive new state, Update Bayesian Probs
                    todo, compute new Bayesian Probs：p(z|x) * Bel(x t-1), update pzx and BayeProb
                    fixme, observe z with 366 dim,
                """

                # reach source
                if self.walker.pos_x == self.src_pos_x and self.walker.pos_z == self.src_pos_z:
                    print("get source")
                    DONE = True
                    r = 10
                    s_ = np.array([0 for u in range(self.n_features)])[np.newaxis, :]

                else:
                    s_ = self.walker.observe_gcc_vector(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                    self.update_bayesia()
                    print(self.Pzx)
                    print(self.BayeProb)
                    s_ = np.array(s_)[np.newaxis, :]

                    # use guide based on Prob
                    # if self.BayeProb[1] > 0.95 and self.detect_which_room() != 1:
                    #     print("guide to Room 1")
                    #     self.walker.reset_walker_pos(-2.0, 1, -2.0)
                    #     s = s_
                    #     a_his = a
                    #     continue

                    print("x: " + str(self.walker.pos_x))
                    print("z: " + str(self.walker.pos_z))

                    # todo, design reward feedback, [explore + entropy + angle_diff]
                    # pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)
                    #
                    # max_angle = max(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))
                    # min_angle = min(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))
                    #
                    # diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)
                    #
                    # r = 1 - diff / 180
                    # r -= (visit[pos_key]) * 0.2

                    # when walker is in room 4, aim to out room, ok and can converge
                    if self.detect_which_room() == 4 or (self.walker.pos_x == 2 and self.walker.pos_z == 1):
                        path_temp = self.walker.find_shortest_path(self.walker.pos_x, self.walker.pos_z,
                                                                   self.room_gates[3][0],
                                                                   self.room_gates[3][2])
                        dis = len(path_temp) - 1

                        # r = 3 - dis * (self.BayeProb[1] + self.BayeProb[2] + self.BayeProb[3])
                        r = 1 - dis * (self.BayeProb[1] + self.BayeProb[2] + self.BayeProb[3]) / 3

                    # when walker is in room 1, aim to reach sound, ok and can converge
                    elif self.detect_which_room() == 1 or (self.walker.pos_x == -2 and self.walker.pos_z == -1):
                        max_angle = max(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))
                        min_angle = min(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))

                        diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)

                        r = 1 - diff / 180

                        pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)
                        r -= (visit[pos_key]) * 0.2

                    # todo, adjust actions in hall and other rooms
                    else:
                        # todo, if set room with low prob to be false ?
                        ROOM[4] = False
                        ROOM[2] = False
                        for i in range(1, 5):
                            path_temp = self.walker.find_shortest_path(self.walker.pos_x, self.walker.pos_z,
                                                                       self.room_gates[i - 1][0],
                                                                       self.room_gates[i - 1][2])
                            locals()['dis%d' % i] = len(path_temp) - 1

                        sum_dis = 0.0
                        for i in range(1, 5):
                            sum_dis += locals()['dis%d' % i] * self.BayeProb[i]

                        # todo, reward should be diff for large distance, i.e., value of 4 need be careful

                        if sum_dis >= 4:
                            addition = 10
                        else:
                            addition = 0
                        r = 1 - (sum_dis + addition) / 4

                        # if self.detect_which_room() == 2:
                        #     r = -0.5
                        #
                        # if self.walker.pos_x == 3.0 and self.walker.pos_z == 0:
                        #     r = 0.1
                        # if self.walker.pos_x == 2.0 and self.walker.pos_z == 0:
                        #     r = 0.5
                        # if self.walker.pos_x == 1.0 and self.walker.pos_z == 0:
                        #     r = 0.7
                        # if self.walker.pos_x == 0.0 and self.walker.pos_z == 0:
                        #     r = 0.8
                        # if self.walker.pos_x == -1.0 and self.walker.pos_z == 0:
                        #     r = 0.9
                        # if self.walker.pos_x == -2.0 and self.walker.pos_z == 0:
                        #     r = 0.9
                        # if self.walker.pos_x == -3.0 and self.walker.pos_z == 0:
                        #     r = 0.6

                    # if self.detect_which_room() == 0:
                    # for i in range(1, 5):
                    #     path_temp = self.walker.find_shortest_path(self.walker.pos_x, self.walker.pos_z,
                    #                                                self.room_gates[i - 1][0],
                    #                                                self.room_gates[i - 1][2])
                    #     locals()['dis%d' % i] = len(path_temp) - 1
                    #
                    # sum_dis = 0.0
                    # # todo, need calculate for all grids in hall to get max num
                    # max_dis = 12
                    #
                    # for i in range(1, 5):
                    #     if ROOM[i] is None:
                    #         sum_dis += locals()['dis%d' % i] * self.BayeProb[i]
                    #
                    # print(sum_dis)
                    #
                    # # todo, reward should be diff for large distance
                    # if sum_dis >= 5:
                    #     addition = 10
                    # else:
                    #     addition = 0
                    #
                    # r = 1 - (sum_dis + addition) / max_dis

                    print("reward: " + str(r))

                """
                    Learn part, ready for next step
                """
                # todo, use multiple td_error backward
                self.walker.learn(s, a, s_, r)
                sum_reward += r
                a_his = a
                s = s_

                if DONE:
                    break


if __name__ == '__main__':
    game = Game()
    # print (game.detect_invalids(3, 1, 0, room1=False))
    game.play()

    # key = str(float(2)) + "_" + str(1) + "_" + str(float(-1))
    #
    # simu = open('env_hole_vol.pkl', 'rb')
    # exp = pickle.load(simu)
    # real = exp[key]
    # print(np.average(real))
    # simu.close()
    #
    # for i in range(1, 5):
    #     simu = open('simu_r%d_vol.pkl' % i, 'rb')
    #     exp = pickle.load(simu)
    #     r_exp = exp[key]
    #
    #     print("room: %d" % i)
    #     print(np.average(r_exp))
    #
    #     simu.close()
