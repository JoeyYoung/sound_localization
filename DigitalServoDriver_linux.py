#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DigitalServoDriver.py
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/15 21:55   msliu      1.0         None
"""

import serial.tools.list_ports


class DigitalServoDriver(object):

    def __init__(self, baud_rate=57600, left_right=1):
        self.baud_rate = baud_rate
        self.port_name, self.port_list = self.detect_serials("USB-Serial Controller")

        if left_right == 1:
            self.left = self.port_name[0]
            self.right = self.port_name[1]
        else:
            self.left = self.port_name[1]
            self.right = self.port_name[0]

        print("The left port is %s, the right port is %s. The baudrate is %d" % (self.left, self.right, self.baud_rate))
        return

    def detect_serials(self, description, vid=0x10c4, pid=0xea60):
        ports = serial.tools.list_ports.comports()
        port_cnt = 0
        port_list = []
        port_path = []
        for port in ports:
            self.print_serial(port)

            # 这有问题，如果两个串口如何检测，如何区分左右

            if port.description.__contains__(description):
                port_list.append(port.description)
                port_path.append(port.device)

        return port_path, port_list

    def print_serial(self, port):
        print("---------------[ %s ]---------------" % port.name)
        print("Path: %s" % port.device)
        print("Descript: %s" % port.description)
        print("HWID: %s" % port.hwid)
        if not None == port.manufacturer:
            print("Manufacture: %s" % port.manufacturer)
        if not None == port.product:
            print("Product: %s" % port.product)
        if not None == port.interface:
            print("Interface: %s" % port.interface)
        print()

    def simple_control_test(self):
        """

        :return:
        """

        return

    pass


if __name__ == '__main__':
    dsd = DigitalServoDriver()
