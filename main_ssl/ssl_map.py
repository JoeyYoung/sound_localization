"""
    hard-encoded corresponding 2D map of CYC 4th floor
"""

import math
import numpy as np
from ssl_setup import STEP_SIZE


class Map:
    def __init__(self):
        # start position
        # mass center of the walker
        self.walker_pos_x = None
        self.walker_pos_z = None

        # world axis indicate walker head
        self.walker_face_to = None

        # max length of walker, safe distance
        self.walker_length = 1.3

        # determine regions and gates
        self.gate_region_1 = [3.2, 7.5]
        self.gate_region_2 = [0, 0.9]
        self.gate_region_3 = [3.2, 0.9]
        self.gate_region_4 = [0.8, 0]

        self.hall_r2_r1 = [0]
        self.hall_r2_r4 = [0, 0, 0]
        self.hall_same = [45, 315, 0]
        self.hall_r3_r1 = [0, 0, 0, 45, 45, 0, 0]

    # just show next position and its facing direction
    def next_walker_pos(self, direction):
        move_towards = (self.walker_face_to + direction) % 360

        x = None
        z = None

        if move_towards == 0:
            x = self.walker_pos_x
            z = self.walker_pos_z + STEP_SIZE
        elif move_towards == 45:
            x = self.walker_pos_x + (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z + (STEP_SIZE * math.sqrt(0.5))
        elif move_towards == 90:
            x = self.walker_pos_x + STEP_SIZE
            z = self.walker_pos_z
        elif move_towards == 135:
            x = self.walker_pos_x + (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z - (STEP_SIZE * math.sqrt(0.5))
        elif move_towards == 180:
            x = self.walker_pos_x
            z = self.walker_pos_z - STEP_SIZE
        elif move_towards == 225:
            x = self.walker_pos_x - (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z - (STEP_SIZE * math.sqrt(0.5))
        elif move_towards == 270:
            x = self.walker_pos_x - STEP_SIZE
            z = self.walker_pos_z
        elif move_towards == 315:
            x = self.walker_pos_x - (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z + (STEP_SIZE * math.sqrt(0.5))
        else:
            print("Fail to cal next position: wrong direction")
            exit(1)

        return x, z, move_towards

    # update position
    def update_walker_pos(self, direction):
        x, z, d = self.next_walker_pos(direction)
        self.walker_pos_x = x
        self.walker_pos_z = z
        self.walker_face_to = d

    # return the set of invalid directions (degrees)
    def detect_invalid_directions(self):
        x = self.walker_pos_x
        z = self.walker_pos_z

        potential_dirs = [0, 45, 90, 135, 180, 225, 270, 315]

        invalids = []

        if 6.0 < z <= 7.5:
            # for dire in potential_dirs:
            #     if (dire + self.walker_face_to) % 360 in [315, 0, 45]:
            #         invalids.append(dire)

            if x < self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [225, 270, 315]:
                        invalids.append(dire)

            if 3.2 <= x:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 45, 135, 180, 225, 315]:
                        invalids.append(dire)

        elif 1.8 < z <= 6.0:
            if x < self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [225, 270, 315]:
                        invalids.append(dire)
            elif x > 3.2 - self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [45, 90, 135]:
                        invalids.append(dire)

        elif 0 <= z <= 1.8:
            if x < 0 or x > 3.2:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 45, 135, 180, 225, 315]:
                        invalids.append(dire)

            if 0 <= x < 1.7:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [135, 225, 315]:
                        invalids.append(dire)

            if 1.7 <= x <= 3.2:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [135, 180, 225]:
                        invalids.append(dire)

        elif z < 0:
            if x < 1.7:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 45, 90, 135]:
                        invalids.append(dire)
            if x > 1.9:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 225, 270, 315]:
                        invalids.append(dire)

        else:
            print("Out of condition for z .")

        return invalids

    # Hall - 0, out_room - 1, left - 2, right - 3, lab - 4, cvlab - 5
    def detect_which_region(self):
        x = self.walker_pos_x
        z = self.walker_pos_z

        current_region = None
        if 0 <= x <= 3.2 and 0 <= z <= 7.5:
            print("Detect walker in Region 0 .")
            current_region = 0
        elif 3.2 < x and 6.0 <= z <= 7.5:
            print("Detect walker in Region 1 .")
            current_region = 1
        elif x < 0 and 0 <= z <= 1.8:
            print("Detect walker in Region 2 .")
            current_region = 2
        elif 3.2 < x and 0 <= z <= 1.8:
            print("Detect walker in Region 3 .")
            current_region = 3
        elif x <= 1.7 and z < 0:
            print("Detect walker in Region 4 .")
            current_region = 4
        elif x >= 3.2 and z < 0:
            print("Detect walker in Region 5 .")
            current_region = 5
        else:
            print("Fail to detect walker region .")

        return current_region

    def cal_distance_region(self, region_num):
        if region_num == 1:
            return np.abs(self.gate_region_1[0] - self.walker_pos_x) + np.abs(self.gate_region_1[1] - self.walker_pos_z)

        elif region_num == 2:
            return np.abs(self.gate_region_2[0] - self.walker_pos_x) + np.abs(self.gate_region_2[1] - self.walker_pos_z)

        elif region_num == 3:
            return np.abs(self.gate_region_3[0] - self.walker_pos_x) + np.abs(self.gate_region_3[1] - self.walker_pos_z)

        elif region_num == 4:
            return np.abs(self.gate_region_4[0] - self.walker_pos_x) + np.abs(self.gate_region_4[1] - self.walker_pos_z)

        else:
            print("no such distance to region %d" % region_num)

    def print_walker_status(self):
        print("walker at x: ", self.walker_pos_x)
        print("walker at z: ", self.walker_pos_z)
        print("walker face to: ", self.walker_face_to)
