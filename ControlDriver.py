#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ControlDriver.py
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2019/10/16 21:46   msliu      1.0

@Description
------------
接受一个线速度v和角速度w
根据线速度和角速度，结算实际转速(RPM-Revolutions Per Minute)


怎么隐隐约约觉得这好像是一个互斥问题啊

"""

import os, sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(BASE_DIR)

import time
from threading import Thread
import DigitalServoDriver_linux as DsD
import serial
import math


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class ControlDriver(Thread):

    def __init__(self, radius_wheel=85.00, flag_end=0, radius=0, left_right=0):
        # radius_wheel = 52.55
        Thread.__init__(self)
        driver = DsD.DigitalServoDriver(left_right=left_right)
        self.left_right = left_right
        baud_rate = driver.baud_rate
        self.ser_r = serial.Serial(driver.right, baud_rate, timeout=None)
        self.ser_l = serial.Serial(driver.left, baud_rate, timeout=None)
        self.radius_wheel = radius_wheel
        self.flag_end = flag_end
        self.radius = radius
        self.speed = 0.1
        self.omega = 0

    def get_rpm_byte(self, rpm):
        rpm_byte = [0x06, 0x00, 0x88, 0x8e]
        # print(rpm)
        rpm_hex = int(rpm / 6000 * 16384)
        # print(rpm_hex)
        if rpm_hex >= 0:
            rpm = [(rpm_hex & 0xFF00) >> 8, (rpm_hex & 0x00FF)]
        else:
            temp = 0xFFFF
            rpm_hex = temp + rpm_hex
            rpm = [(rpm_hex & 0xFF00) >> 8, (rpm_hex & 0x00FF)]
        rpm_byte[1] = rpm[0]
        rpm_byte[2] = rpm[1]
        rpm_byte.pop(3)
        last = 0
        for item in rpm_byte:
            last = last + item
        if last > 256:
            last = last & 0xFF
        rpm_byte.append(last)
        return rpm_byte

    def get_speed_rpm(self, w):
        # ?first version
        # rpm = (self.speed + w) / (2 * math.pi * self.radius_wheel / 1000) * 60
        rpm = w / (2 * math.pi * self.radius_wheel / 1000) * 60
        # print(int(rpm))
        return int(rpm)

    def get_rpm_Omega(self):
        """
        r * w = v = l (vr + vl)
                    -----------
                    2 (vr - vl)
        :return:
        """
        if self.omega > 0:
            vl = (self.radius + (56 / 2)) / 100 * self.omega
            vr = (self.radius - (56 / 2)) / 100 * self.omega
        else:
            vl = -(self.radius - (56 / 2)) / 100 * self.omega
            vr = -(self.radius + (56 / 2)) / 100 * self.omega

        return vl, vr

    def control_part(self):
        print("start control part")
        start = [0x00, 0x00, 0x01, 0x01]
        pc_mode = [0x02, 0x00, 0xc4, 0xc6]
        rpm = [0x06, 0x00, 0x88, 0x8e]
        end = [0x00, 0x00, 0x00, 0x00]
        self.ser_l.write(bytes(start))
        self.ser_r.write(bytes(start))
        self.ser_l.write(bytes(pc_mode))
        self.ser_r.write(bytes(pc_mode))

        while True:
            vl, vr = self.get_rpm_Omega()
            # print("Omega: %f %f" %( vl, vr))
            # print("Speed: %f " % self.speed)
            if self.left_right == 1:
                left = self.get_rpm_byte(self.get_speed_rpm(vl) + self.get_speed_rpm(self.speed))
                right = self.get_rpm_byte(-(self.get_speed_rpm(vr) + self.get_speed_rpm(self.speed)))
            else:
                # print((self.get_speed_rpm(vl) + self.get_speed_rpm(self.speed)))
                left = self.get_rpm_byte((self.get_speed_rpm(vl) + self.get_speed_rpm(self.speed)))
                right = self.get_rpm_byte(-(self.get_speed_rpm(vr) + self.get_speed_rpm(self.speed)))
            # print(left, right)
            self.ser_l.write(bytes(left))
            self.ser_r.write(bytes(right))
            time.sleep(0.3)
            watch = [0x80, 0x00, 0x80]
            self.ser_l.write(bytes(watch))
            self.ser_r.write(bytes(watch))

            if self.flag_end != 0:
                break

        self.ser_l.write(bytes(end))
        self.ser_r.write(bytes(end))
        time.sleep(20)

        return

    def run(self):
        self.control_part()

    pass


if __name__ == '__main__':
    def loop(cd):
        time.sleep(100)
        cd.flag_end = 1


    cd = ControlDriver()
    t1 = Thread(target=loop, args=(cd,))

    cd.start()
    t1.start()
