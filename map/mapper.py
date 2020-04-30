# Sound Source locate
#
# @Time    : 2019-03-10
# @Author  : xyzhao
# @File    : mapper.py
# @Description: to process pgm file using opencv

import cv2
import numpy as np
import os


# 255 - white, 0 - black
# res: 1 pixel - 0.02m

class Mapper:
    def __init__(self):
        self.img_file = "CB4120311.pgm"

    """
        For CB example, return 1088 * 1088 2d array
    """

    def get_2d_map(self, toshow=False):
        print(" Processing map figure ...")

        img = cv2.imread(self.img_file)
        print("original size", img.size)

        # median filter, remove some extra points
        img_medi = cv2.medianBlur(img, 3, 0)
        print("Medi filter size", img_medi.size)

        # extract edge information
        # focus on gradient, (min, max), > max as edge point, < min drop
        canny = cv2.Canny(img_medi, 100, 200)
        print("canny size", canny.size)

        # do smooth, so that the wall can be seen as entity
        img_gaus = cv2.GaussianBlur(canny, (5, 5), 0)
        print("Gaus size", img_gaus.size)

        # obtain list of contours
        img_, contours, hierarchy = cv2.findContours(img_gaus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # turn white ground and black point
        _, img_ = cv2.threshold(img_, 1, 255, cv2.THRESH_BINARY_INV)

        img_cots = cv2.drawContours(img_, contours, -1, 0, 1)

        print("img_contours size", img_cots.size)

        print("height : weight", img_cots.shape[0], img_cots.shape[1])

        # reshape to a 1088 x 1088 figure
        img_cut = np.zeros([1088, 1088])
        for i in range(1088):
            for j in range(1088):
                img_cut[i][j] = img_cots[i][j + 88]

        # adjust window size
        if toshow is True:
            cv2.namedWindow("contour", cv2.WINDOW_NORMAL)
            cv2.imshow("contour", img_cut)

            cv2.waitKey()

        return img_cut


# todo, pixels are too large, use tree to represent the map, the unit is related to the size of walker

class QuadNode:
    def __init__(self, radius, parent):
        # size of this block
        self.radius = radius

        # link to parent
        self.parent = parent

        # four child
        self.children = [None] * 4

        # four map corners
        self.corners = [None] * 4

        # to indicate occupied or not, 1/0
        self.occupied = None


def build_tree(node):
    if node.radius < 20:
        return

    # add children nodes
    num0_child = QuadNode(node.radius / 2, node)
    num0_child.corners = [node.corners[0],
                          [node.corners[1][0], (node.corners[1][1] + node.corners[0][1]) / 2],
                          [(node.corners[2][0] + node.corners[0][0]) / 2, node.corners[2][1]],
                          [(node.corners[3][0] + node.corners[0][0]) / 2,
                           (node.corners[3][1] + node.corners[0][1]) / 2]]

    num1_child = QuadNode(node.radius / 2, node)
    num1_child.corners = [[node.corners[1][0], (node.corners[1][1] + node.corners[0][1]) / 2],
                          node.corners[1],
                          [(node.corners[3][0] + node.corners[0][0]) / 2,
                           (node.corners[3][1] + node.corners[0][1]) / 2],
                          [(node.corners[1][0] + node.corners[3][0]) / 2, node.corners[1][1]]]

    num2_child = QuadNode(node.radius / 2, node)
    num2_child.corners = [[(node.corners[0][0] + node.corners[2][0]) / 2, node.corners[0][1]],
                          [(node.corners[3][0] + node.corners[0][0]) / 2,
                           (node.corners[3][1] + node.corners[0][1]) / 2],
                          node.corners[2],
                          [node.corners[2][0], (node.corners[2][1] + node.corners[3][1]) / 2]]

    num3_child = QuadNode(node.radius / 2, node)
    num3_child.corners = [[(node.corners[3][0] + node.corners[0][0]) / 2,
                           (node.corners[3][1] + node.corners[0][1]) / 2],
                          [(node.corners[1][0] + node.corners[3][0]) / 2, node.corners[3][1]],
                          [node.corners[3][0], (node.corners[2][1] + node.corners[3][1]) / 2],
                          node.corners[3]]

    node.children = [num0_child, num1_child, num2_child, num3_child]

    # each children node, call build tree
    for i in node.children:
        build_tree(i)


def read_points(inputdirs):
    res = []

    for inputdir in inputdirs:
        files = os.listdir(inputdir)
        for file in files:
            # skip dir
            name = str(file.title()).lower()

            if os.path.isdir(file) or name[:6] != 'points':
                continue

            with open(os.path.join(inputdir, file), 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    xy = line.split(";")
                    temp = [float(xy[0]), float(xy[1])]
                    res.append(temp)
    return res


if __name__ == '__main__':
    # mapper = Mapper()
    # img = np.array(mapper.get_2d_map(toshow=True))
    #
    # unit = img.shape[0]
    #
    # for i in range(unit):
    #     for j in range(unit):
    #         img[i][j] = 255

    unit = 1000
    img = np.zeros([unit, unit])

    for i in range(unit):
        for j in range(unit):
            img[i][j] = 255

    dirs = ["./test_step1"]
    points = read_points(dirs)

    for point in points:
        x = int(point[0] * 25)
        y = int(point[1] * 25)

        if x > 1000 or y > 1000:
            continue
        img[x][y] = 0

    cv2.namedWindow("contour", cv2.WINDOW_NORMAL)
    cv2.imshow("contour", img)

    cv2.waitKey()

    # build tree root
    # root = QuadNode(unit, None)
    # root.corners = [[0, 0], [0, root.radius], [root.radius, 0], [root.radius, root.radius]]
    #
    # build_tree(root)

    # todo, set not occupied node set as leaf,
