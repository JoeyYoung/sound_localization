# Sound Source locate
#
# @Time    : 2019-10-9 19:03
# @Author  : xyzhao
# @File    : generateGcc.py
# @Description: process wav file into features

import numpy as np
import math
import pickle
import wave
import collections
import os
import random
import copy
import sys
import matplotlib.pyplot as plt

'''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
'''


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    if isinstance(sig, list):
        sig = np.array(sig)

    if isinstance(refsig, list):
        refsig = np.array(refsig)

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift  # / float(interp * fs) * 340

    return tau, cc


"""
    walk all fires in input dir:
        collections: pos -> [gcc features, label]
        meet new walker pos:  read 4 mics, process; read label, process (01 vector)
        meet old walker pos:  continue
    
    store ifo to file in the format :   [[features], [label]]
                                            ...
                                        [[features], [label]]
    
    note: output file has property (room, source pos)
"""


# 已经把几号位的麦克风和信道对应好了
def generate_gcc_real(input_dir, output_dir, output_file):
    res = collections.defaultdict(list)
    gcc_width_half = 30
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        if os.path.isdir(file) or name[:2] != 'u_':
            continue

        file_names = name.split('_')
        pos = file_names[2] + "_" + file_names[3] + "_" + file_names[5]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            temp = int(file_names[5])

            index_fill = int(temp / 45)
            label = [0] * 8
            label[index_fill] = 1

            # read 4 mirs, compute features

            min_len = 999999
            fs = 0

            # i indicates 几号位
            for i in range(1, 5):
                if i == 1:
                    j = 2
                elif i == 2:
                    j = 4
                elif i == 3:
                    j = 3
                elif i == 4:
                    j = 1
                else:
                    j = 0

                mic = name[:len(name) - 5] + str(j) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)

                if len(data) < min_len:
                    min_len = len(data)

                locals()['data%d' % i] = data

            gcc_vector = []

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            for i in range(1, 5):
                for j in range(i + 1, 5):
                    tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                    for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                        gcc_vector.append(cc[k])
            res[pos] = [gcc_vector, label]

        print(len(res.keys()))  # 1088

    # write into file

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)


# collect real data using walker
def generate_gcc_deploy(input_dir, output_dir, output_file):
    res = collections.defaultdict(list)
    gcc_width_half = 30
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        if os.path.isdir(file) or name[:4] != 'real':
            continue

        file_names = name.split('_')
        pos = file_names[1] + "_" + file_names[2]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            temp = int(file_names[2])

            index_fill = int(temp / 45)
            label = [0] * 8
            label[index_fill] = 1

            # read 4 mirs, compute features

            min_len = 999999
            fs = 0

            # i indicates 几号位
            for i in range(1, 5):
                mic = name[:len(name) - 5] + str(i) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)

                if len(data) < min_len:
                    min_len = len(data)

                locals()['data%d' % i] = data

            gcc_vector = []

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            for i in range(1, 5):
                for j in range(i + 1, 5):
                    tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                    for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                        gcc_vector.append(cc[k])
            res[pos] = [gcc_vector, label]

        print(len(res.keys()))  # 1088

    # write into file

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)


# rsc back simulated env
# data[1] represents right
def generate_gcc_simu_rscback(input_dir, output_dir, output_file):
    res = collections.defaultdict(list)
    gcc_width_half = 30
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        if os.path.isdir(file) or name[:3] != 'src':
            continue

        file_names = name.split('_')
        pos = file_names[1] + "_" + file_names[2] + "_" + file_names[4]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            temp = int(file_names[5])

            index_fill = int(temp / 45)
            label = [0] * 8
            label[index_fill] = 1

            # read 4 mirs, compute features

            min_len = 999999
            fs = 0

            # i indicates 几号位, 需要选择不一样的wav文件的特定信道
            for i in range(1, 5):

                mic = name[:len(name) - 5] + str(i) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)

                if len(data) % 2 != 0:
                    data = np.append(data, 0)
                data.shape = -1, 2
                data = data.T

                # data[0] represents right, data[1] represents left

                data_pro = data[1]
                # [(data[0][j] + data[1][j]) / 2 for j in range(len(data[0]))]

                if len(data_pro) < min_len:
                    min_len = len(data_pro)

                locals()['data%d' % i] = data_pro

            gcc_vector = []

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            for i in range(1, 5):
                for j in range(i + 1, 5):
                    tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                    for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                        gcc_vector.append(cc[k])
            res[pos] = [gcc_vector, label]

        print(len(res.keys()))  # 1088

    # write into file

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)


def generate_gcc(input_dir, output_dir, output_file, average=True, vector=True, savepos=False):
    res = collections.defaultdict(list)
    gcc_width_half = 30
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        if os.path.isdir(file) or name[:6] != 'walker':
            continue

        file_names = name.split('_')
        pos = file_names[1] + "_" + file_names[2] + "_" + file_names[3]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            index_fill = int(int(file_names[4]) / 45)
            label = [0] * 8
            label[index_fill] = 1

            # read 4 mirs, compute features

            min_len = 999999
            fs = 0

            for i in range(1, 5):
                mic = name[:len(name) - 5] + str(i) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                if average is True:
                    data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
                    if len(data) % 2 != 0:
                        data = np.append(data, 0)
                    data.shape = -1, 2
                    data = data.T

                    data_avg = [(data[0][j] + data[1][j]) / 2 for j in range(len(data[0]))]
                    if len(data_avg) < min_len:
                        min_len = len(data_avg)

                    locals()['data%d' % i] = data_avg

                else:
                    data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
                    if len(data) < min_len:
                        min_len = len(data)
                    locals()['data%d' % i] = data

            gcc_offset = []
            gcc_vector = []

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            if vector is True:
                for i in range(1, 5):
                    for j in range(i + 1, 5):
                        tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                        for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                            gcc_vector.append(cc[k])
                res[pos] = [gcc_vector, label]

            else:
                for i in range(1, 5):
                    for j in range(i + 1, 5):
                        tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                        gcc_offset.append(tau)

                res[pos] = [gcc_offset, label]

        print(len(res.keys()))  # 1088

    # write into file

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)

    # fixme, save variable res into disk (binary)
    if savepos is True:
        disk = open('simu_r2.pkl', 'wb')
        pickle.dump(res, disk)
        disk.close()


def generate_srp(input_dir, output_dir, output_file, average=True, vector=True, savepos=False):
    res = collections.defaultdict(list)
    gcc_width_half = 60
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        if os.path.isdir(file) or name[:6] != 'walker':
            continue

        file_names = name.split('_')
        pos = file_names[1] + "_" + file_names[2] + "_" + file_names[3]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            index_fill = int(int(file_names[4]) / 45)
            label = [0] * 8
            label[index_fill] = 1

            # read 4 mirs, compute features

            min_len = 999999
            fs = 0

            for i in range(1, 5):
                mic = name[:len(name) - 5] + str(i) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                if average is True:
                    data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
                    if len(data) % 2 != 0:
                        data = np.append(data, 0)
                    data.shape = -1, 2
                    data = data.T

                    data_avg = [(data[0][j] + data[1][j]) / 2 for j in range(len(data[0]))]
                    if len(data_avg) < min_len:
                        min_len = len(data_avg)

                    locals()['data%d' % i] = data_avg

                else:
                    data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
                    if len(data) < min_len:
                        min_len = len(data)
                    locals()['data%d' % i] = data

            gcc_offset = []
            gcc_vector = []
            cc_srp = None

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            if vector is True:
                for i in range(1, 5):
                    for j in range(1, 5):
                        tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                        if cc_srp is None:
                            cc_srp = cc
                        else:
                            cc_srp += cc

                for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                    gcc_vector.append(cc_srp[k])

                res[pos] = [gcc_vector, label]

            else:
                for i in range(1, 5):
                    for j in range(i + 1, 5):
                        tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                        gcc_offset.append(tau)

                res[pos] = [gcc_offset, label]

        print(len(res.keys()))  # 1088

    # write into file

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)

    # fixme, save variable res into disk (binary)
    if savepos is True:
        disk = open('new.pkl', 'wb')
        pickle.dump(res, disk)
        disk.close()


def generate_gcc_binary(input_dir, output_dir, output_file):
    room1_pos = [-2.0, 1.6, -1.0]
    room2_pos = [2.0, 1.6, -1.0]
    room3_pos = [-2.0, 1.6, 1.0]
    room4_pos = [2.0, 1.6, 1.0]

    room1_x = [i for i in np.arange(-3.5, 0, 0.5)]
    room1_z = [i for i in np.arange(-4.5, -1, 0.5)]

    room2_x = [i for i in np.arange(0.5, 4.0, 0.5)]
    room2_z = [i for i in np.arange(-4.5, -1, 0.5)]

    room3_x = [i for i in np.arange(-3.5, 0, 0.5)]
    room3_z = [i for i in np.arange(1.5, 5.0, 0.5)]

    room4_x = [i for i in np.arange(0.5, 4.0, 0.5)]
    room4_z = [i for i in np.arange(1.5, 5.0, 0.5)]

    hall_x = [i for i in np.arange(-3.5, 4.0, 0.5)]
    hall_z = [i for i in np.arange(-0.5, 1.0, 0.5)]

    res = collections.defaultdict(list)
    gcc_width_half = 30
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        if os.path.isdir(file) or name[:6] != 'walker':
            continue

        file_names = name.split('_')
        pos = file_names[1] + "_" + file_names[2] + "_" + file_names[3]

        if res.get(pos) is not None:
            continue
        else:
            min_len = 999999
            fs = 0

            label = [0] * 2
            # fixme, add binary label based on in/out door, need indicate room type
            # if (float(file_names[1]) in hall_x and float(file_names[3]) in hall_z) or (
            #         float(file_names[1]) == room1_pos[0] and float(file_names[3]) == room1_pos[2]) or (
            #         float(file_names[1]) == room2_pos[0] and float(file_names[3]) == room2_pos[2]) or (
            #         float(file_names[1]) == room3_pos[0] and float(file_names[3]) == room3_pos[2]) or (
            #         float(file_names[1]) == room4_pos[0] and float(file_names[3]) == room4_pos[2]):
            #     label = [1, 0]
            # else:
            #     label = [0, 1]
            label = [1, 0]

            for i in range(1, 5):
                mic = name[:len(name) - 5] + str(i) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
                if len(data) % 2 != 0:
                    data = np.append(data, 0)
                data.shape = -1, 2
                data = data.T

                data_avg = [(data[0][j] + data[1][j]) / 2 for j in range(len(data[0]))]
                if len(data_avg) < min_len:
                    min_len = len(data_avg)

                locals()['data%d' % i] = data_avg

            gcc_vector = []

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            for i in range(1, 5):
                for j in range(i + 1, 5):
                    tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                    for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                        gcc_vector.append(cc[k])

            print("======")
            print(pos)
            print(label)
            res[pos] = [gcc_vector, label]

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)


def split_test_tain(dir, unionfile, testfire, trainfire, test_size, train_size):
    temp = [i for i in range(1, test_size + train_size + 1)]
    random.shuffle(temp)
    index_test = temp[:test_size]

    with open(os.path.join(dir, unionfile), 'r+') as f:
        lines = f.readlines()
        test_lines = []
        train_lines = []
        print(len(lines))

        count = 0
        for i in range(len(lines)):
            index = i + 1
            if index in index_test:
                test_lines.append(lines[i])
            else:
                count += 1
                train_lines.append(lines[i])

        with open(os.path.join(dir, testfire), 'w+') as t:
            t.writelines(test_lines)
        with open(os.path.join(dir, trainfire), 'w+') as n:
            n.writelines(train_lines)

        print(count)


def cal_volume(waveData, frameSize, overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen * 1.0 / step))
    volume = np.zeros((frameNum, 1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
        curFrame = curFrame - np.median(curFrame)  # zero-justified
        volume[i] = np.sum(np.abs(curFrame))
    return volume


# fixme, generate strength features used for reward; store res[pos] = [gcc, label], can be look up
def generate_volume(input_dir, output_dir, output_file):
    res = {}
    files = os.listdir(input_dir)
    done = 0
    for file in files:
        # skip dir
        name = str(file.title()).lower()
        if os.path.isdir(file) or name[:6] != 'walker':
            continue
        file_names = name.split('_')
        pos = file_names[1] + "_" + file_names[2] + "_" + file_names[3]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            # read 4 mirs, compute volume
            frameSize = 256
            overLap = 128

            min_len = 999999

            for i in range(1, 5):
                mic = name[:len(name) - 5] + str(i) + ".wav"
                fw = wave.open(os.path.join(input_dir, mic), 'r')
                params = fw.getparams()
                nchannels, sampwidth, framerate, nframes = params[:4]
                strData = fw.readframes(nframes)
                waveData = np.fromstring(strData, dtype=np.int16)
                waveData = waveData * 1.0 / max(abs(waveData))  # normalization
                fw.close()
                if len(waveData) < min_len:
                    min_len = len(waveData)

                locals()['data%d' % i] = waveData

            vol = []
            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]
                waveData = locals()['data%d' % i]

                volumes = cal_volume(waveData, frameSize, overLap)
                vol.append(np.average(volumes[300:3600]))

                res[pos] = vol

        done += 1
        print(done)

    with open(os.path.join(output_dir, output_file), 'w') as f:
        for k in res.keys():
            print(res[k], file=f)

    # fixme, save variable res into disk (binary)
    disk = open('env_hole_vol.pkl', 'wb')
    pickle.dump(res, disk)
    disk.close()


def generate_obs(input_dir, output):
    res = collections.defaultdict(list)
    gcc_width_half = 30
    # the whole vector length is 61

    files = os.listdir(input_dir)
    for file in files:
        # skip dir
        name = str(file.title()).lower()

        file_names = name.split('_')
        pos = file_names[0]

        # meet old walker pos
        if res.get(pos) is not None:
            continue
        else:
            # read 4 mirs, compute features

            min_len = 999999
            fs = 0

            for i in range(1, 5):
                mic = name[:len(name) - 5] + str(i) + ".wav"
                wav = wave.open(os.path.join(input_dir, mic), 'rb')

                n_frame = wav.getnframes()
                fs = wav.getframerate()

                data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)
                if len(data) % 2 != 0:
                    data = np.append(data, 0)
                data.shape = -1, 2
                data = data.T

                data_avg = [(data[0][j] + data[1][j]) / 2 for j in range(len(data[0]))]
                if len(data_avg) < min_len:
                    min_len = len(data_avg)

                locals()['data%d' % i] = data_avg

            gcc_vector = []

            for i in range(1, 5):
                locals()['data%d' % i] = locals()['data%d' % i][:min_len]

            for i in range(1, 5):
                for j in range(i + 1, 5):
                    tau, cc = gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                    for k in range(min_len - gcc_width_half, min_len + gcc_width_half + 1):
                        gcc_vector.append(cc[k])
            res[pos] = gcc_vector
            print(pos)

    disk = open(output, 'wb')
    pickle.dump(res, disk)
    disk.close()


if __name__ == '__main__':
    # fixme, change output file and gcc type here, rl env no need split
    # wavedir = './wavdata/multiple/add/bin_src_1_1.6_2'
    # gccdir = './gccdata/multiple/eight_classific/'
    # gccdir = './gccdata/multiple/hole_eight'
    # srpdir = './srpdata'
    #
    # split_test_tain(gccdir, unionfile='vector_hole', testfire='vector_test', trainfire='vector_train',
    #                 test_size=4, train_size=3100)

    # pickle dump res[pos] = [gcc, label] into env.pkl
    # generate_gcc_binary(wavedir, gccdir, output_file='label_add')
    # generate_gcc(wavedir, gccdir, output_file='label_add', average=True, vector=True, savepos=False)
    # split_test_tain(gccdir, unionfile='vector_eight', testfire='vector_test', trainfire='vector_train',
    #                 test_size=400, train_size=2033)
    # generate_volume(wavedir, gccdir, output_file='volume')

    # generate_obs('./wavdata/multiple/add/0_0', '0_0.pkl')

    # a = './wavdata/8x3x8_src_-2_1.6_-2'
    # b = './wavdata/8x3x8_src_-2_1.6_2'
    # c = './wavdata/8x3x8_src_-3_1.6_0'
    # d = './wavdata/8x3x8_src_0_1.6_-1.5'
    # e = './wavdata/8x3x8_src_0_1.6_0'
    # f = './wavdata/8x3x8_src_0_1.6_3'
    # g = './wavdata/8x3x8_src_2_1.6_-2'
    # h = './wavdata/8x3x8_src_2_1.6_2'
    # i = './wavdata/8x3x8_src_3.5_1.6_-3.5'
    #
    # j = './wavdata/multiple/eight_classific/src_-2_1.6_-3'
    # k = './wavdata/multiple/eight_classific/src_-2_1.6_3'
    # l = './wavdata/multiple/eight_classific/src_0_1.6_0'
    # m = './wavdata/multiple/eight_classific/src_2_1.6_-3'
    # n = './wavdata/multiple/eight_classific/src_2_1.6_3'
    #
    # wavs = [a, b, c, d, e, f, g, h, i, j, k, l, m, n]
    #
    # srpdir = './srpdata'
    #
    # count = 0
    # for filename in wavs:
    #     generate_srp(filename, srpdir, output_file='label_' + str(count), average=True, vector=True, savepos=False)
    #     count += 1
    #     print("========== %d" % count)

    # audiodir = './audio'
    # gccdir = './audio'
    # generate_gcc_real(audiodir, gccdir, output_file='label_real', average=True, vector=True, savepos=False)

    """
        Generate data for training
    """
    # a = './wavdata/hole/src_-1_3'
    # b = './wavdata/hole/src_1_4'
    # c = './wavdata/hole/src_-2_0'
    # d = './wavdata/hole/src_-2_-2'
    # e = './wavdata/hole/src_-3_-4'
    #
    # wavs = [e]
    #
    # count = 4
    # gccdir = './gccdata/multiple/hole_eight'
    # for dirname in wavs:
    #     generate_gcc(dirname, gccdir, output_file='vector_' + str(count), average=True, vector=True, savepos=False)
    #     count += 1
    #     print("========== %d" % count)

    """
        generate part for training and exp
    """
    # voldir = './voldata/hole'
    # wavdir3 = './wavdata/hole/src_-2_3_exp'
    # wavdir4 = './wavdata/hole/src_2_3_exp'
    # wavdir2 = './wavdata/hole/src_2_-3_exp'
    # wavdir1 = './wavdata/hole/src_-2_-3_exp'
    # wavdir5 = './wavdata/hole/src_-2_-4_rl'
    #
    # # todo, modify pickle file name - simu_r%d
    # generate_gcc(wavdir2, gccdir, output_file='exp_' + '2', average=True, vector=True, savepos=True)
    # generate_volume(wavdir5, voldir, output_file='rl_vol')

    #
    #
    # """
    #     generate rl to pickle only
    # """
    # # todo, modify pickle file name, no use label file for training

    # generate_gcc(wavdir5, gccdir, output_file='rl', average=True, vector=True, savepos=True)

    # for i in range(11, 12):
    #     wavedir = './wav/rsc_back_wo_diff/src%d' % i
    #     gccdir = './gcc/rsc_back_wo_diff'
    #     generate_gcc_simu_rscback(wavedir, gccdir, 'src%d' % i)

    # wavedir = './wav/real_cyc4'
    gccdir = './gcc/cyc4'
    # generate_gcc_deploy(wavedir, gccdir, 'cyc4')

    split_test_tain(gccdir, unionfile='cyc4', testfire='cyc4_test', trainfire='cyc4_train', test_size=10, train_size=189)
