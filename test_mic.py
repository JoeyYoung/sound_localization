import os
import sys

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import pyaudio
import wave
from scipy.io import wavfile

CHUNK = 1024
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
RECORD_WIDTH = 2
CHANNELS = 4
RATE = 16000
RECORD_SECONDS = 3
ACTION_SECONDS = 4
FORMAT = pyaudio.paInt16


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

# stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 input_device_index=device_index)
