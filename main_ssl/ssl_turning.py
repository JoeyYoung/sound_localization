"""
    SSL _ turning
"""
import math
import time


def SSLturning(cd, angle):
    time_sleep_value = 0.05
    cd.speed = 0
    cd.omega = 0
    cd.radius = 0
    # cd: an instance of class ControlandOdometryDriver,  angle: angle to turn as in degree
    # angle = 0, 45, 90, 135, 180, 225, 270, 315
    if angle > 180:
        rad = (360 - angle) / 180 * math.pi
    else:
        rad = -angle / 180 * math.pi

    currentTHETA = cd.position[2]  # read current THETA∈(-π，π]
    expectedTHETA = currentTHETA + rad

    if expectedTHETA > math.pi:
        expectedTHETA -= 2 * math.pi
    elif expectedTHETA <= -math.pi:
        expectedTHETA += 2 * math.pi

    # print('rad: ', rad, ';  Current theta: ', currentTHETA, '; Expected theta: ', expectedTHETA)

    if rad != 0:
        if rad > 0:
            cd.omega = math.pi / 6
        else:
            cd.omega = - math.pi / 6
        cd.radius = 0
        cd.speed = 0
        time.sleep(time_sleep_value)
        # print('start moving...')

        while 1:
            if (cd.position[2] * expectedTHETA) > 0:
                break

        if (cd.position[2] * expectedTHETA) >= 0 and rad > 0:
            while 1:
                if abs(cd.position[2] - expectedTHETA) <= 0.2:
                    cd.omega = 0
                    time.sleep(time_sleep_value)
                    # print('reached')
                    break
        elif (cd.position[2] * expectedTHETA) >= 0 and rad < 0:
            while 1:
                if abs(expectedTHETA - cd.position[2]) <= 0.2:
                    cd.omega = 0
                    time.sleep(time_sleep_value)
                    # print('reached')
                    break
        else:
            print('false')
            pass
    else:
        pass

    cd.omega = 0
    time.sleep(0.1)
    # print('final position: ', cd.position[2])
