# Sound Source locate
# 
# @Time    : 2019-11-01 16:36
# @Author  : xyzhao
# @File    : read_from_mic.py
# @Description: record and split wav files from real microphones

import pyaudio
import wave
import json
import signal
import sys
import os
import time
import numpy as np

"""
    {'index': 0, 
    'structVersion': 2, 
    'name': 'USB Camera-B4.09.24.1', 
    'hostApi': 0, 
    'maxInputChannels': 4, 
    'maxOutputChannels': 0, 
    'defaultLowInputLatency': 0.00725, 
    'defaultLowOutputLatency': 0.01, 
    'defaultHighInputLatency': 0.03525, 
    'defaultHighOutputLatency': 0.1, 
    'defaultSampleRate': 16000.0}
"""

RECORD_RATE = 16000
RECORD_CHANNELS_DEFAULT = 1
RECORD_CHANNELS = 4

# sample width is 2 bytes
RECORD_WIDTH = 2
CHUNK = 1024
RECORD_SECONDS = 1

RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
WAVE_NAME = "./records_mic/test.wav"
wave_file = wave.open(WAVE_NAME, "wb")

p = pyaudio.PyAudio()

buffer1 = list(range(CHUNK))
buffer2 = list(range(CHUNK))
buffer3 = list(range(CHUNK))
buffer4 = list(range(CHUNK))


def open_files():
    wave_file.setnchannels(RECORD_CHANNELS)
    wave_file.setsampwidth(2)
    wave_file.setframerate(RECORD_RATE)


def close_files():
    wave_file.close()


def sigint_handler(signum, frame):
    stream.stop_stream()
    stream.close()
    p.terminate()
    close_files()
    print('catched interrupt signal!')
    sys.exit(0)


if __name__ == "__main__":
    # Register ctrl-c interruption
    # signal.signal(signal.SIGINT, sigint_handler)

    print("Number of devices: ", p.get_device_count())

    device_index = -1
    stream = None

    # scan to get usb device
    for index in range(0, p.get_device_count()):
        info = p.get_device_info_by_index(index)
        device_name = info.get("name")
        print("device_name: ", device_name)

        # find mic usb device
        if device_name.find(RECORD_DEVICE_NAME) != -1:
            device_index = index
            break

    if device_index != -1:
        print("find the device")

        print(p.get_device_info_by_index(device_index))

        stream = p.open(
            rate=RECORD_RATE,
            format=p.get_format_from_width(RECORD_WIDTH),
            channels=RECORD_CHANNELS,
            input=True,
            input_device_index=device_index,
            start=True,
            frames_per_buffer=CHUNK)
    else:
        print("don't find the device")

    # fixme, begin recording
    open_files()

    time.sleep(2)

    stream.start_stream()
    print("* recording")

    for i in range(0, int(RECORD_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        print("length of data: %d" % (len(data)))
        print(max(data))

        for j in range(CHUNK):
            # assert((data[j*8] | (data[j*8 + 1] << 8)) == data[j*8]+data[j*8+1]*256)
            # print("%x" %(data[j*8] | (data[j*8 + 1] << 8)),
            #	  "\t%x %x" %(data[j*8 + 2], data[j*8 + 3]),
            #	  "\t%x %x" % (data[j*8 + 4], data[j*8 + 5]),
            #	  "\t%x %x" % (data[j*8 + 6], data[j*8 + 7])
            #	  )

            # bytes_buffer1 = bytes_buffer1 + data[j*8 + 0]
            # bytes_buffer1[j*2 + 1] = data[j*8 + 1]
            # bytes_buffer1[j*2 + 0] = data[j*8 + 2]
            # bytes_buffer1[j*2 + 1] = data[j*8 + 3]
            # bytes_buffer1[j*2 + 0] = data[j*8 + 4]
            # bytes_buffer1[j*2 + 1] = data[j*8 + 5]
            # bytes_buffer1[j*2 + 0] = data[j*8 + 6]
            # bytes_buffer1[j*2 + 1] = data[j*8 + 7]

            buffer1[j] = data[j * 8 + 0] | (data[j * 8 + 1] << 8)
            buffer2[j] = data[j * 8 + 2] | (data[j * 8 + 3] << 8)
            buffer3[j] = data[j * 8 + 4] | (data[j * 8 + 5] << 8)
            buffer4[j] = data[j * 8 + 6] | (data[j * 8 + 7] << 8)
            if j == 0 and i == 0:
                print("%x\t%x\t%x\t%x" % (buffer1[j], buffer2[j], buffer3[j], buffer4[j]))

        wave_file.writeframes(data)
    # wave_file1.writeframes(bytes_buffer1)
    # wave_file2.writeframes(bytes_buffer2)
    # wave_file3.writeframes(bytes_buffer3)
    # wave_file4.writeframes(bytes_buffer4)

    print("* done recording")
    stream.stop_stream()
    close_files()
    # audio_data should be raw_data
    print("record end")
