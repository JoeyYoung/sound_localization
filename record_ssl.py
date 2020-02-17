"""
    always read, until wake up, choose part to compute gcc
"""

import pyaudio
import wave
from scipy.io import wavfile
import tensorflow as tf
import numpy as np
import sys
import os
import math
import time
import collections
import threading
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import DigitalDriver.ControlDriver as CD

"""
    Record Parameters
"""

CHUNK = 1024
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
RECORD_WIDTH = 2
CHANNELS = 4
RATE = 16000
RECORD_SECONDS = 3
FORMAT = pyaudio.paInt16

FORWARD_SECONDS = 2

WAV_PATH = "../resource/wav/"


def read_wav(file):
    wav = wave.open(file, 'rb')
    fn = wav.getnframes()  # 207270
    fr = wav.getframerate()  # 44100
    fw = wav.getsampwidth()  # 44100
    f_data = wav.readframes(fn)
    data = np.frombuffer(f_data, dtype=np.short)
    return data


def cal_volume(waveData, frameSize=256, overLap=128):
    waveData = waveData * 1.0 / max(abs(waveData))  # normalization
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen * 1.0 / step))
    volume = np.zeros((frameNum, 1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
        curFrame = curFrame - np.median(curFrame)  # zero-justified
        volume[i] = np.sum(np.abs(curFrame))
    return volume


def split_channels(wave_output_filename):
    sampleRate, musicData = wavfile.read(wave_output_filename)
    mic1 = []
    mic2 = []
    mic3 = []
    mic4 = []
    for item in musicData:
        mic1.append(item[0])
        mic2.append(item[1])
        mic3.append(item[2])
        mic4.append(item[3])

    front = wave_output_filename[:len(wave_output_filename) - 4]

    # physic mic number --- channel number
    wavfile.write(front + '_mic1.wav', sampleRate, np.array(mic2))
    wavfile.write(front + '_mic2.wav', sampleRate, np.array(mic3))
    wavfile.write(front + '_mic3.wav', sampleRate, np.array(mic1))
    wavfile.write(front + '_mic4.wav', sampleRate, np.array(mic4))


def judge_active(wave_output_filename):
    sampleRate, musicData = wavfile.read(wave_output_filename)
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    for item in musicData:
        d1.append(item[0])
        d2.append(item[1])
        d3.append(item[2])
        d4.append(item[3])

    v1 = np.average(np.abs(d1))
    v2 = np.average(np.abs(d2))
    v3 = np.average(np.abs(d3))
    v4 = np.average(np.abs(d4))

    threshold_v = 230

    if v1 > threshold_v or v2 > threshold_v or v3 > threshold_v or v4 > threshold_v:
        print(v1, v2, v3, v4)
        return True
    else:
        return False


def loop_record(control):
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

    saved_count = 0

    # steps
    while True:
        """
            Record
        """

        # active detection
        while True:
            print("start monitoring ... ")
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index)

            # 16 data
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            print("End monitoring ... ")

            # temp store into file
            wave_output_filename = "real_" + str(saved_count) + "_" + sys.argv[1] + ".wav"
            wf = wave.open(os.path.join(WAV_PATH, wave_output_filename), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(RECORD_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # if exceed, break, split to process, then action. After action done, begin monitor
            if judge_active(os.path.join(WAV_PATH, wave_output_filename)) is True:
                break

        """
            Split
        """
        split_channels(os.path.join(WAV_PATH, wave_output_filename))

        """
            use four mic file to be input to produce action
        """

        print("apply movement ...")

        time.sleep(FORWARD_SECONDS)

        print("movement done.")

        # begin next step
        saved_count += 1


if __name__ == '__main__':
    # cd = Control()
    # loop_record(cd)

    cd = CD.ControlDriver()
    p1 = threading.Thread(target=loop_record, args=(cd,))
    p2 = threading.Thread(target=cd.control_part, args=())
    print("hehe")
    p2.start()
    p1.start()
