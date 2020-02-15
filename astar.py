# Sound Source locate
# 
# @Time    : 2019-10-25 14:49
# @Author  : xyzhao
# @File    : astar.py
# @Description: avoid obstacle module


import numpy as np


class Node:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.G = 0
        self.H = 0
        self.F = self.G + self.H
        self.parent = None

        # fixme, indicate step length
        self.unit = 1.0
        self.neighs_pos = [[x, z - self.unit], [x + self.unit, z - self.unit],
                           [x + self.unit, z], [x + self.unit, z + self.unit],
                           [x, z + self.unit], [x - self.unit, z + self.unit],
                           [x - self.unit, z], [x - self.unit, z - self.unit]]

        self.neighs = None

    def set_neighs(self):
        self.neighs = [Node(n[0], n[1]) for n in self.neighs_pos]

    def set_parent(self, par):
        self.parent = par

    def set_H(self, destination):
        self.H = abs(destination.x - self.x) + abs(destination.z - self.z)

    def set_G(self, G):
        self.G = G

    def isInList(self, list):
        for n in list:
            if self.x == n.x and self.z == n.z:
                return True
        return False


class Astar:
    def __init__(self):
        # build grids map
        # fixme, indicate step length
        self.unit = 1.0

        # fixme, indicate env and obstacles
        self.grids_x = [i for i in np.arange(-3.0, 3.0 + self.unit, self.unit)]
        self.grids_z = [i for i in np.arange(-4.0, 4.0 + self.unit, self.unit)]

        # set obstacles
        self.wall_axis_z = {-4: [i for i in np.arange(-5.0, 6.0, 1.0)],
                            4: [i for i in np.arange(-5.0, 6.0, 1.0)],
                            0: [i for i in np.arange(-5.0, 6.0, 1.0) if i != 0]}
        self.wall_axis_x = {5: [i for i in np.arange(-4.0, 5.0, 1.0)],
                            1: [i for i in np.arange(-4.0, 5.0, 1.0) if i != -2 and i != 2],
                            -1: [i for i in np.arange(-4.0, 5.0, 1.0) if i != -2 and i != 2],
                            -5: [i for i in np.arange(-4.0, 5.0, 1.0)]}

        self.close_list = None
        self.open_list = None

    # fixme, should indicate source and destination
    def find_path(self, x1, z1, x2, z2):
        if x1 == x2 and z1 == z2:
            return [[x1, z1]]

        start = Node(x1, z1)
        destination = Node(x2, z2)
        path = []

        self.close_list = []
        self.open_list = []

        self.open_list.append(start)

        DONE = False
        while DONE is False:
            # find node in open list with min F as current
            # todo, need optimize to find min, using heap
            min_F = 99999
            min_node = None
            for node in self.open_list:
                if node.F < min_F:
                    min_F = node.F
                    min_node = node

            current = min_node

            # remove it from open_list to close_list
            self.open_list.remove(current)
            self.close_list.append(current)

            # scan its neighbors
            current.set_neighs()
            for node in current.neighs:
                # fixme, detect invalids
                if self.wall_axis_x.get(node.z) is not None:
                    if node.x in self.wall_axis_x[node.z]:
                        continue
                if self.wall_axis_z.get(node.x) is not None:
                    if node.z in self.wall_axis_z[node.x]:
                        continue

                # node in close list, has been visited
                if node.isInList(self.close_list):
                    continue

                # node new, compute F, G, H, update par
                if node.isInList(self.open_list) is False:
                    self.open_list.append(node)
                    node.set_parent(current)
                    node.set_H(destination)

                    # here set cost for each step to be 1
                    node.set_G(current.G + 1)

                # node not new, need update par if G is smaller
                else:
                    if current.G + 1 < node.G:
                        node.set_G(current.G + 1)
                        node.set_parent(current)

            # DONE = True
            if destination.isInList(self.open_list):
                DONE = True
                path.append([destination.x, destination.z])
                while current is not None:
                    path.append([current.x, current.z])
                    current = current.parent

                path.reverse()

        # todo, get action list based on path
        return path


if __name__ == '__main__':
    a = Astar()
    p = a.find_path(2.0, 3.0, -3.0, -4.0)
    print(p)
