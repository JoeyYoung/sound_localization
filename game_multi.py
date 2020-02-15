# Sound Source locate
# 
# @Time    : 2019-10-23 12:43
# @Author  : xyzhao
# @File    : game_multi.py
# @Description: online learning in multiple rooms

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
        self.max_epoch = 30
        self.max_steps = 100

        # define sound source information
        # fixme
        self.src_pos_x = -3.0
        self.src_pos_y = 1.6
        self.src_pos_z = -4.0

        # sample as a grid map with 0.5m unit
        # fixme, change step length to 1m
        self.unit = 1.0
        self.room_grids_x = [i for i in np.arange(-3.0, 3.0 + self.unit, self.unit)]
        self.room_grids_z = [i for i in np.arange(-4.0, 4.0 + self.unit, self.unit)]

        # fixme, define wall and obstacles
        self.wall_axis_z = {-4: [i for i in np.arange(-5.0, 6.0, 1.0)],
                            4: [i for i in np.arange(-5.0, 6.0, 1.0)],
                            0: [i for i in np.arange(-5.0, 6.0, 1.0) if i != 0]}
        self.wall_axis_x = {5: [i for i in np.arange(-4.0, 5.0, 1.0)],
                            1: [i for i in np.arange(-4.0, 5.0, 1.0) if i != -2 and i != 2],
                            -1: [i for i in np.arange(-4.0, 5.0, 1.0) if i != -2 and i != 2],
                            -5: [i for i in np.arange(-4.0, 5.0, 1.0)]}

        # fixme, define checkpoints: room gates, hall center
        self.room_gates = [[-2.0, 1, -1.0], [2.0, 1, -1.0], [-2.0, 1, 1.0], [2.0, 1, 1.0]]
        self.hall_center = [[0, 0, 0]]

        # fixme, define room zone
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

        # todo, add some obstacles
        return invalids

    # fixme, return 1, 2, 3, 4 room, 0-hall
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
        based on guide path to learn actions:
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

    def play(self):
        records_step = []
        records_r = []

        """
            Begin epoch
        """
        for epoch in range(self.max_epoch):
            print("========== Epoch %d ======" % epoch)
            memory = collections.defaultdict(dict)
            visit = {}
            for i in self.room_grids_x:
                for j in self.room_grids_z:
                    visit[str(i) + "*" + str(j)] = 0
                    for k in self.walker.action_labels:
                        memory[str(i) + "*" + str(j)][k] = 0

            # init walker position
            # fixme, random choose
            self.walker.reset_walker_pos(2.0, 1, 3.0)
            DONE = False

            sum_reward = 0.0

            a_his = None

            # fixme, lock room zone and room gates
            ROOM = [None] * 5

            """
                Begin steps
            """
            for step in range(self.max_steps):
                print("************** step %d" % step)
                GUIDE = False

                print("x: " + str(self.walker.pos_x))
                print("z: " + str(self.walker.pos_z))

                s = self.walker.observe_gcc_vector(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                s = np.array(s)[np.newaxis, :]

                # fixme, judge: if walker in room, out or in.
                room_type = self.detect_which_room()

                # fixme, if already determine this room, not go out
                if ROOM[room_type] is not True:
                    # walker in room
                    if room_type in [1, 2, 3, 4]:
                        print("detect walker in room%d " % room_type)
                        # source is not in the room, GUIDE

                        # todo, give more obs about binary
                        if self.walker.sound_in_room(s) is False:
                            print("source is not in room%d" % room_type)

                            path = self.walker.find_shortest_path(self.walker.pos_x, self.walker.pos_z,
                                                                  self.room_gates[int("%d" % room_type) - 1][0],
                                                                  self.room_gates[int("%d" % room_type) - 1][2])

                            self.walker.reset_walker_pos(self.room_gates[int("%d" % room_type) - 1][0],
                                                         self.walker.pos_y,
                                                         self.room_gates[int("%d" % room_type) - 1][2])
                            print("guide to room gate %d " % room_type)

                            if room_type == 1 or room_type == 2:
                                self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                             self.walker.pos_z + self.unit)
                            else:
                                self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                             self.walker.pos_z - self.unit)

                            print("step further to the hall ")

                            # fixme, based on path generate experiences to learn
                            self.learn_guide_actions(path, visit)

                            ROOM[room_type] = False

                            GUIDE = True

                        # source in the room
                        else:
                            print("find source in room %d" % room_type)
                            ROOM[room_type] = True
                            HALL = False

                    # walker in the gate, GUIDE into room
                    elif room_type == -1:
                        f = 0
                        if self.walker.pos_x == self.room_gates[0][0] and self.walker.pos_z == self.room_gates[0][2]:
                            self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                         self.walker.pos_z - self.unit)
                            f = 1

                        elif self.walker.pos_x == self.room_gates[1][0] and self.walker.pos_z == self.room_gates[1][2]:
                            self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                         self.walker.pos_z - self.unit)
                            f = 2

                        elif self.walker.pos_x == self.room_gates[2][0] and self.walker.pos_z == self.room_gates[2][2]:
                            self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                         self.walker.pos_z + self.unit)
                            f = 3

                        elif self.walker.pos_x == self.room_gates[3][0] and self.walker.pos_z == self.room_gates[3][2]:
                            self.walker.reset_walker_pos(self.walker.pos_x, self.walker.pos_y,
                                                         self.walker.pos_z + self.unit)
                            f = 4

                        print("detect walker in gate%d" % f)
                        print("step further into room%d" % f)
                        GUIDE = True

                    elif room_type == 0:
                        print("detect walker in the hall")
                        # todo, give more obs when walker in the hall

                    # else: walker in the hall
                    # if step == 0 \
                    #         or [self.walker.pos_x, self.walker.pos_y, self.walker.pos_z] in self.room_gates \
                    #         or [self.walker.pos_x, self.walker.pos_y, self.walker.pos_z] in self.hall_center:
                    #     fe = open('first_obs.pkl', 'rb')
                    #     obs = pickle.load(fe)
                    #
                    #     # ==================== Right obs
                    #     s_r = obs['right']
                    #     s_r = np.array(s_r)[np.newaxis, :]
                    #     a_r, p_r = self.walker.choose_action(s_r, [])
                    #     p_rr = [p_r[len(p_r) - 2], p_r[len(p_r) - 1]]
                    #     p_rr = np.append(p_rr, p_r[:len(p_r) - 2])
                    #
                    #     # ==================== Left obs
                    #     s_l = obs['left']
                    #     s_l = np.array(s_l)[np.newaxis, :]
                    #     a_l, p_l = self.walker.choose_action(s_l, [])
                    #     p_ll = [p_l[0], p_l[1]]
                    #     p_ll = np.append(p_l[2:], p_ll)
                    #
                    #     # ==================== Down obs
                    #     s_d = obs['down']
                    #     s_d = np.array(s_d)[np.newaxis, :]
                    #     a_d, p_d = self.walker.choose_action(s_d, [])
                    #     p_dd = [p_d[len(p_d) - 4], p_d[len(p_d) - 3], p_d[len(p_d) - 2], p_d[len(p_d) - 1]]
                    #     p_dd = np.append(p_dd, p_d[:len(p_d) - 4])
                    #
                    #     # ==================== Decide action
                    #     p_mix = [0] * self.n_actions
                    #     for i in range(self.n_actions):
                    #         if i not in invalids:
                    #             p_mix[i] = p[i] + p_rr[i] + p_ll[i] + p_dd[i]
                    #
                    #     p_mix = np.array(p_mix)
                    #     p_mix /= p_mix.sum()
                    #     a_mix = np.argmax(p_mix)
                    #
                    #     fe.close()
                    #
                    #     a = a_mix
                    #     a_his = a
                    #     p = p_mix
                    #     direction = self.walker.action_labels[a]

                    # if walker is guided to a new pos

                if GUIDE is True:
                    # fixme, init a_his if guide to a new pos
                    a_his = None
                    continue

                # detect walls and obstacles
                invalids = self.detect_invalids(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z, ROOM)

                # fixme, cut down action space, but for the hall part allow more
                pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)
                for i in memory[pos_key].keys():
                    if self.detect_which_room() == 0:
                        threshold = 5
                    else:
                        threshold = 2
                    if memory[pos_key][i] >= threshold:
                        invalids.append(self.walker.action_labels.index(i))
                visit[pos_key] += 1

                a, p = self.walker.choose_action(s, invalids)
                if a_his is None:
                    a_his = a

                direction = self.walker.action_labels[a]

                # print(p)
                print(direction)

                memory[pos_key][direction] += 1

                # step next
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
                    # r = 1 - abs((a + a_his) % self.n_actions - a_his) / (self.n_actions - 1)
                    pos_key = str(self.walker.pos_x) + "*" + str(self.walker.pos_z)

                    max_angle = max(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))
                    min_angle = min(float(self.walker.action_labels[a]), float(self.walker.action_labels[a_his]))

                    diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)

                    r = 1 - diff / 180
                    r -= (visit[pos_key]) * 0.2

                    # # note action has been performed
                    # # fixme, give additional reward when in hall
                    if self.detect_which_room() == 0:
                        for i in range(1, 5):
                            path_temp = self.walker.find_shortest_path(self.walker.pos_x, self.walker.pos_z,
                                                                       self.room_gates[i - 1][0],
                                                                       self.room_gates[i - 1][2])
                            locals()['dis%d' % i] = len(path_temp) - 1

                        sum_dis = 0.0
                        # todo, need calculate for all grids in hall to get max num
                        max_dis = 12

                        for i in range(1, 5):
                            if ROOM[i] is None:
                                sum_dis += locals()['dis%d' % i]

                        # todo, reward should be diff for large distance
                        if sum_dis >= 10:
                            addition = 10
                        else:
                            addition = 0

                        r = 1 - (sum_dis + addition) / max_dis

                    # todo, give punishment when step into false Room
                    # will only first step to gate, then inner room guide until to hall
                    if self.walker.pos_x == 2 and self.walker.pos_z == -2:
                        r -= 1
                    if self.walker.pos_x == 2 and self.walker.pos_z == -1:
                        r -= 1

                    print("x: " + str(self.walker.pos_x))
                    print("z: " + str(self.walker.pos_z))
                    print("reward: " + str(r))
                    # give punishment if detect obstacles
                    # pub = self.detect_invalids(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                    # if len(pub) > 0:
                    #     r -= 0.5

                    s_ = self.walker.observe_gcc_vector(self.walker.pos_x, self.walker.pos_y, self.walker.pos_z)
                    s_ = np.array(s_)[np.newaxis, :]

                sum_reward += r
                a_his = a

                self.walker.learn(s, a, s_, r)
                if DONE:
                    break

            # evaluate epoch
            print("-----------------------------------" + str(sum_reward / step))
            # records_step.append(step)
            # records_r.append(sum_reward / step)


if __name__ == '__main__':
    game = Game()
    # print (game.detect_invalids(3, 1, 0, room1=False))
    game.play()
    # print(game.walker.observe_gcc_vector(0.0, 1, 0.0)
    # s = game.walker.observe_gcc_vector(-1.0, 1, -4.0)
    # s = np.array(s)[np.newaxis, :]
    # print(game.walker.sound_in_room(s))

    # env = open('0_0.pkl', 'rb')
    # tz = pickle.load(env)
    # env.close()
    #
    # invalids2 = [1, 3, 4, 5, 7]
    # invalids0 = [0, 1, 3, 4, 5, 7]
    #
    # s = game.walker.observe_gcc_vector(-3.0, 1, 0.0)
    # s = np.array(s)[np.newaxis, :]
    # p = game.walker.choose_action(s, [])
    # print(p)
    #
    # # ================= Right obs
    # print("right ... ")
    # s_r = tz['right']
    # s_r = np.array(s_r)[np.newaxis, :]
    # a_r, p_r = game.walker.choose_action(s_r, [])
    # p_rr = [p_r[len(p_r) - 2], p_r[len(p_r) - 1]]
    # p_rr = np.append(p_rr, p_r[:len(p_r) - 2])
    # print(p_rr)
    #
    # # ==================== Left obs
    # print("left ... ")
    # s_l = tz['left']
    # s_l = np.array(s_l)[np.newaxis, :]
    # a_l, p_l = game.walker.choose_action(s_l, [])
    # p_ll = [p_l[0], p_l[1]]
    # p_ll = np.append(p_l[2:], p_ll)
    # print(p_ll)
