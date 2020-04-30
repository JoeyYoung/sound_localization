import pyaudio
import ctypes as ct
import numpy as np
import wave
import math
import matplotlib.pyplot as plt
import pyaudio

device_index = -1

p = pyaudio.PyAudio()

"""
    Recognize Mic device, before loop
"""
# scan to get usb device
print(p.get_device_count())
for index in range(0, p.get_device_count()):
    info = p.get_device_info_by_index(index)
    device_name = info.get("name")
    print("device_name: ", device_name)

    # find mic usb device
    if device_name.find(RECORD_DEVICE_NAME) != -1:
        device_index = index
        # break

if device_index != -1:
    print("find the device")

    print(p.get_device_info_by_index(device_index))
else:
    print("don't find the device")

