"""
    analyze, capture useful gcc features.

"""

import wave
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

file = '/Users/xyzhao/PycharmProjects/supervised/wavdata/8x3x8_src_0_1.6_0/walker_0.0_1_1.0_0_mic1.wav'
# file = '/Users/xyzhao/PycharmProjects/supervised/wavdata/multiple/multi_src_-2_1.6_-3/walker_2.0_1_3.0_0_mic1.wav'
# file = '/Users/xyzhao/PycharmProjects/supervised/wavdata/multiple/eight_classific/src_-2_1.6_-3/r1_mic1.wav'
# file = '/Users/xyzhao/PycharmProjects/supervised/wavdata/multiple/eight_classific/src_-2_1.6_-3/walker_-3.0_1_-4.0_135_mic1.wav'

print("======== use wave to read")
wav = wave.open(file, 'rb')

n_frame = wav.getnframes()  # 获取帧数 207270
fs = wav.getframerate()
n_channel = wav.getnchannels()

f_data = wav.readframes(n_frame)
data = np.frombuffer(f_data, dtype=np.short)  # 621810

if len(data) % 2 != 0:
    data = np.append(data, 0)
data.shape = -1, 2
data = data.T

data_avg = [(data[0][j] + data[1][j]) / 2 for j in range(len(data[0]))]

print(len(data_avg))  # 16000

# print("======= use wavfile to read")

# sampleRate, musicData = wavfile.read(file)
# print(sampleRate)
# print(len(musicData))   # 47104
#
#
# time = np.arange(0, len(data)) * (1.0 / fs)
# unit = 1
# b_time = []
# b_data2 = []
# sum_2 = 0


time = np.arange(0, len(data_avg)) * (1.0 / fs)
unit = 2000
b_time = []
b_data2 = []
sum_2 = 0
for i in range(len(time)):
    if i % unit == 0:
        b_time.append(time[i])
        b_data2.append(sum_2 / unit)
        sum_2 = 0
    sum_2 += data_avg[i]

plt.cla()
plt.plot(b_time, b_data2)
plt.show()
