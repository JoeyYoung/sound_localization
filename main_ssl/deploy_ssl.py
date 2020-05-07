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
import ctypes as ct

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import DigitalDriver.ControlandOdometryDriver as CD

"""
    Record Parameters
"""

GCC_LENG = 366
GCC_BIAS = 6
ACTION_SPACE = 8
CHUNK = 1024
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
RECORD_WIDTH = 2
CHANNELS = 4
RATE = 16000

RECORD_SECONDS = 1
FORMAT = pyaudio.paInt16

FORWARD_SECONDS = 3
STEP_SIZE = 1

MODEL_PATH = "../resource/model/save20.ckpt"
WAV_PATH = "../resource/wav/online"
ONLINE_MODEL_PATH = "../resource/model/online.ckpt"

"""
    Digital Driver Part
"""

"""
    Map
"""


# corresponding 2D map

class Map:
    def __init__(self):
        # start position
        # mass center of the walker
        self.walker_pos_x = None
        self.walker_pos_z = None

        # world axis indicate walker head
        self.walker_face_to = None

        # max length of walker, safe distance
        self.walker_length = 1.3

        # determine regions and gates
        self.gate_region_1 = [3.2, 7.5]
        self.gate_region_2 = [0, 0.9]
        self.gate_region_3 = [3.2, 0.9]
        self.gate_region_4 = [0.8, 0]

        self.hall_r2_r1 = [0]
        self.hall_r2_r4 = [0, 0, 0]
        self.hall_same = [45, 315, 0]
        self.hall_r3_r1 = [0, 0, 0, 45, 45, 0, 0]

    # just show next position and its facing direction
    def next_walker_pos(self, direction):
        move_towards = (self.walker_face_to + direction) % 360

        x = None
        z = None

        if move_towards == 0:
            x = self.walker_pos_x
            z = self.walker_pos_z + STEP_SIZE
        elif move_towards == 45:
            x = self.walker_pos_x + (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z + (STEP_SIZE * math.sqrt(0.5))
        elif move_towards == 90:
            x = self.walker_pos_x + STEP_SIZE
            z = self.walker_pos_z
        elif move_towards == 135:
            x = self.walker_pos_x + (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z - (STEP_SIZE * math.sqrt(0.5))
        elif move_towards == 180:
            x = self.walker_pos_x
            z = self.walker_pos_z - STEP_SIZE
        elif move_towards == 225:
            x = self.walker_pos_x - (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z - (STEP_SIZE * math.sqrt(0.5))
        elif move_towards == 270:
            x = self.walker_pos_x - STEP_SIZE
            z = self.walker_pos_z
        elif move_towards == 315:
            x = self.walker_pos_x - (STEP_SIZE * math.sqrt(0.5))
            z = self.walker_pos_z + (STEP_SIZE * math.sqrt(0.5))
        else:
            print("Fail to cal next position: wrong direction")
            exit(1)

        return x, z, move_towards

    # update position
    def update_walker_pos(self, direction):
        x, z, d = self.next_walker_pos(direction)
        self.walker_pos_x = x
        self.walker_pos_z = z
        self.walker_face_to = d

    # return the set of invalid directions (degrees)
    def detect_invalid_directions(self):
        x = self.walker_pos_x
        z = self.walker_pos_z

        potential_dirs = [0, 45, 90, 135, 180, 225, 270, 315]

        invalids = []

        if 6.0 < z <= 7.5:
            # for dire in potential_dirs:
            #     if (dire + self.walker_face_to) % 360 in [315, 0, 45]:
            #         invalids.append(dire)

            if x < self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [225, 270, 315]:
                        invalids.append(dire)

            if 3.2 <= x:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 45, 135, 180, 225, 315]:
                        invalids.append(dire)

        elif 1.8 < z <= 6.0:
            if x < self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [225, 270, 315]:
                        invalids.append(dire)
            elif x > 3.2 - self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [45, 90, 135]:
                        invalids.append(dire)

        elif 0 <= z <= 1.8:
            if x < 0 or x > 3.2:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 45, 135, 180, 225, 315]:
                        invalids.append(dire)

            if 0 <= x < 1.7:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [135, 225, 315]:
                        invalids.append(dire)

            if 1.7 <= x <= 3.2:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [135, 180, 225]:
                        invalids.append(dire)

        elif z < 0:
            if x < 1.7:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 45, 90, 135]:
                        invalids.append(dire)
            if x > 1.9:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [0, 225, 270, 315]:
                        invalids.append(dire)

        else:
            print("Out of condition for z .")

        return invalids

    # Hall - 0, out_room - 1, left - 2, right - 3, lab - 4, cvlab - 5
    def detect_which_region(self):
        x = self.walker_pos_x
        z = self.walker_pos_z

        current_region = None
        if 0 <= x <= 3.2 and 0 <= z <= 7.5:
            print("Detect walker in Region 0 .")
            current_region = 0
        elif 3.2 < x and 6.0 <= z <= 7.5:
            print("Detect walker in Region 1 .")
            current_region = 1
        elif x < 0 and 0 <= z <= 1.8:
            print("Detect walker in Region 2 .")
            current_region = 2
        elif 3.2 < x and 0 <= z <= 1.8:
            print("Detect walker in Region 3 .")
            current_region = 3
        elif x <= 1.7 and z < 0:
            print("Detect walker in Region 4 .")
            current_region = 4
        elif x >= 3.2 and z < 0:
            print("Detect walker in Region 5 .")
            current_region = 5
        else:
            print("Fail to detect walker region .")

        return current_region

    def cal_distance_region(self, region_num):
        if region_num == 1:
            return np.abs(self.gate_region_1[0] - self.walker_pos_x) + np.abs(self.gate_region_1[1] - self.walker_pos_z)

        elif region_num == 2:
            return np.abs(self.gate_region_2[0] - self.walker_pos_x) + np.abs(self.gate_region_2[1] - self.walker_pos_z)

        elif region_num == 3:
            return np.abs(self.gate_region_3[0] - self.walker_pos_x) + np.abs(self.gate_region_3[1] - self.walker_pos_z)

        elif region_num == 4:
            return np.abs(self.gate_region_4[0] - self.walker_pos_x) + np.abs(self.gate_region_4[1] - self.walker_pos_z)

        else:
            print("no such distance to region %d" % region_num)

    def print_walker_status(self):
        print("walker at x: ", self.walker_pos_x)
        print("walker at z: ", self.walker_pos_z)
        print("walker face to: ", self.walker_face_to)


"""
    GCC Processor Part
"""


class GccGenerator:
    def __init__(self):
        self.gcc_width_half = 30
        self.gcc_width_half_bias = 50

    def gcc_phat(self, sig, refsig, fs=1, max_tau=None, interp=1):
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

    def cal_gcc_online(self, input_dir, save_count, type='Vector', debug=True, denoise=True):
        for i in range(1, 5):
            if debug:
                if i == 1:
                    p = 2
                elif i == 2:
                    p = 4
                elif i == 3:
                    p = 1
                elif i == 4:
                    p = 3
            else:
                p = i

            if denoise is True:
                mic_name = str(save_count) + "_de_" + "mic%d" % p + ".wav"
            else:
                mic_name = str(save_count) + "_" + "mic%d" % p + ".wav"

            wav = wave.open(os.path.join(input_dir, mic_name), 'rb')

            n_frame = wav.getnframes()
            fs = wav.getframerate()
            data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)

            locals()['data%d' % i] = data

        gcc_vector = []

        center = int(len(locals()['data%d' % 1]) / 2)

        gcc_bias = []
        for i in range(1, 5):
            for j in range(i + 1, 5):
                tau, cc = self.gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                for k in range(center - self.gcc_width_half, center + self.gcc_width_half + 1):
                    gcc_vector.append(cc[k])
                gcc_bias.append(cc)

        # add bias
        pair1 = gcc_bias[0]
        pair2 = gcc_bias[1]
        pair3 = gcc_bias[2]
        pair4 = gcc_bias[3]
        pair5 = gcc_bias[4]
        pair6 = gcc_bias[5]

        center = int(len(pair1) / 2)

        p1 = pair1[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p2 = pair2[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p3 = pair3[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p4 = pair4[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p5 = pair5[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]
        p6 = pair6[center - self.gcc_width_half_bias:center + self.gcc_width_half_bias]

        bias1 = list(p1).index(np.max(p1)) - self.gcc_width_half_bias
        bias2 = list(p2).index(np.max(p2)) - self.gcc_width_half_bias
        bias3 = list(p3).index(np.max(p3)) - self.gcc_width_half_bias
        bias4 = list(p4).index(np.max(p4)) - self.gcc_width_half_bias
        bias5 = list(p5).index(np.max(p5)) - self.gcc_width_half_bias
        bias6 = list(p6).index(np.max(p6)) - self.gcc_width_half_bias

        bias = [bias1, bias2, bias3, bias4, bias5, bias6]

        if type == 'Bias':
            return bias

        return gcc_vector


"""
    RL online training Part
"""


class Actor:
    def __init__(self, n_features, n_actions, lr):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='state')  # [1, n_F]
        self.a = tf.placeholder(tf.int32, None, name='action')  # None
        self.td_error = tf.placeholder(tf.float32, None, name='td-error')  # None

        # restore from supervised learning model
        with tf.variable_scope('Supervised'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=int(math.sqrt(self.n_actions * self.n_features)),
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        # define new loss function for actor
        with tf.variable_scope('actor_loss'):
            log_prob = tf.log(self.acts_prob[0, self.a] + 0.0000001)  # self.acts_prob[0, self.a]
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        # when load all variables in, we need reset optimizer
        with tf.variable_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(-self.exp_v)

            self.reset_optimizer = tf.variables_initializer(optimizer.variables())

        self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load_trained_model(self, model_path):
        # when load models, variables are transmit: layers, adam (not placeholder and op)
        self.saver.restore(self.sess, model_path)
        # load l1, acts_prob and adam vars
        self.sess.run(self.reset_optimizer)

    # invalid indicates action index
    def output_action(self, s, invalid_actions):
        acts = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        # mask invalid actions based on invalid actions
        p = acts.ravel()
        p = np.array(p)

        for i in range(self.n_actions):
            if i in invalid_actions:
                p[i] = 0

        # choose invalid action with possible 1
        if p.sum() == 0:
            print("determine invalid action")
            act = np.random.choice(np.arange(acts.shape[1]))
            exit(1)
        else:
            p /= p.sum()
            # act = np.random.choice(np.arange(acts.shape[1]), p=p)
            act = np.argmax(p)

        return act, p

    def learn(self, s, a, td):
        # may modify s
        # s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict=feed_dict)


class Critic:
    def __init__(self, n_features, n_actions, lr, gamma):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.v_ = tf.placeholder(tf.float32, [None, 1], name='v_next')  # [1,1]
        self.r = tf.placeholder(tf.float32, None, name='reward')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=int(math.sqrt(1 * self.n_features)),
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='v'
            )

        with tf.variable_scope('td_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('critic_optimizer'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # global will init actor vars, partly init
        # need init: layer, optimizer (placeholder and op init is unnecessary)
        # self.sess.run(tf.global_variables_initializer())
        uninitialized_vars = [var for var in tf.global_variables() if 'critic' in var.name or 'Critic' in var.name]

        initialize_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(initialize_op)

    def learn(self, s, r, s_):
        # need modify s, s_
        # s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error


class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]


class Float(ct.Union):
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]


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


def nextpow2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1


def de_noise(input_file, output_file):
    f = wave.open(input_file)

    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    fs = framerate
    str_data = f.readframes(nframes)
    f.close()
    x = np.fromstring(str_data, dtype=np.short)

    len_ = 20 * fs // 1000  # 样本中帧的大小
    PERC = 50  # 窗口重叠占帧的百分比
    len1 = len_ * PERC // 100  # 重叠窗口
    len2 = len_ - len1  # 非重叠窗口
    # 设置默认参数
    Thres = 3
    Expnt = 2.0
    beta = 0.002
    G = 0.9

    win = np.hamming(len_)
    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)

    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    nFFT = 2 * 2 ** (nextpow2(len_))
    noise_mean = np.zeros(nFFT)

    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    # --- allocate memory and initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # =========================    Start Processing   ===============================
    for n in range(0, Nframes):
        # Windowing
        insign = win * x[k - 1:k + len_ - 1]
        # compute fourier transform of a frame
        spec = np.fft.fft(insign, nFFT)
        # compute the magnitude
        sig = abs(spec)

        # save the noisy phase information
        theta = np.angle(spec)
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        def berouti(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 4 - SNR * 3 / 20
            else:
                if SNR < -5.0:
                    a = 5
                if SNR > 20:
                    a = 1
            return a

        def berouti1(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 3 - SNR * 2 / 20
            else:
                if SNR < -5.0:
                    a = 4
                if SNR > 20:
                    a = 1
            return a

        if Expnt == 1.0:  # 幅度谱
            alpha = berouti1(SNRseg)
        else:  # 功率谱
            alpha = berouti(SNRseg)
        #############
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt;
        # 当纯净信号小于噪声信号的功率时
        diffw = sub_speech - beta * noise_mu ** Expnt

        # beta negative components

        def find_index(x_list):
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list

        z = find_index(diffw)
        if len(z) > 0:
            # 用估计出来的噪声信号表示下限值
            sub_speech[z] = beta * noise_mu[z] ** Expnt
            # --- implement a simple VAD detector --------------
        if SNRseg < Thres:  # Update noise spectrum
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
            noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
        # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
        # 交换上下对称元素
        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
        x_phase = (sub_speech ** (1 / Expnt)) * (
                    np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
        # take the IFFT

        xi = np.fft.ifft(x_phase).real
        # --- Overlap and add ---------------
        xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1:len_]
        k = k + len2
    # 保存文件
    wf = wave.open(output_file, 'wb')
    # 设置参数
    wf.setparams(params)
    # 设置波形文件 .tostring()将array转换为data
    wave_data = (winGain * xfinal).astype(np.short)
    wf.writeframes(wave_data.tostring())
    wf.close()


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

    threshold_v = 400 # 230

    if v1 > threshold_v or v2 > threshold_v or v3 > threshold_v or v4 > threshold_v:
        print("Voice intensity: ", v1, v2, v3, v4)
        return True
    else:
        return False


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


def loop_record(control, event, source='test'):
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
    gccGenerator = GccGenerator()
    map = Map()

    # fixme, set start position
    map.walker_pos_x = -2.1
    map.walker_pos_z = 0.9
    map.walker_face_to = 90
    # 1.0, 1.85, 0
    # -3.1, 0.9, 90

    actor = Actor(GCC_BIAS, ACTION_SPACE, lr=0.004)
    critic = Critic(GCC_BIAS, ACTION_SPACE, lr=0.003, gamma=0.95)

    # fixme, use oneline model if needed
    actor.load_trained_model(MODEL_PATH)

    # init at the first step
    state_last = None
    action_last = None
    direction_last = None

    # steps
    while True:
        event.wait()
        print("===== %d =====" % saved_count)
        map.print_walker_status()
        map.detect_which_region()

        """
            Record
        """

        # active detection
        print("start monitoring ... ")
        while True:
            # print("start monitoring ... ")
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

            # print("End monitoring ... ")

            # temp store into file
            wave_output_filename = str(saved_count) + ".wav"
            wf = wave.open(os.path.join(WAV_PATH, wave_output_filename), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(RECORD_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # todo, de-noise into new file, then VAD and split
            noise_file = wave_output_filename
            denoise_file = str(saved_count) + "_de.wav"

            de_noise(os.path.join(WAV_PATH, noise_file), os.path.join(WAV_PATH, denoise_file))

            # if exceed, break, split to process, then action. After action done, begin monitor
            if judge_active(os.path.join(WAV_PATH, denoise_file)) is True:
                print("Detected ... ")
                break

        """
            Split
        """
        split_channels(os.path.join(WAV_PATH, denoise_file))

        """
            use four mic file to be input to produce action
        """

        print("producing action ...")

        # fixme, change debug model if mic change
        gcc = gccGenerator.cal_gcc_online(WAV_PATH, saved_count, type='Bias', debug=False)
        state = np.array(gcc)[np.newaxis, :]

        print("GCC Bias :", gcc)

        # todo, define invalids, based on constructed map % restrict regions
        invalids_dire = map.detect_invalid_directions()

        print("invalids_dire of walker: ", invalids_dire)

        # transform walker direction to mic direction
        invalids_idx = [(i + 45) % 360 / 45 for i in invalids_dire]

        print("invalids_idx of mic: ", invalids_idx)

        # set invalids_idx in real test
        action, _ = actor.output_action(state, [])

        print("prob of mic: ", _)

        # transform mic direction to walker direction
        direction = (action + 6) % 7 * 45

        # bias is 45 degree, ok
        print("Estimated direction of walker : ", direction)

        # fixme, for test or hard code, cover direction
        # direction = int(input())
        if source == '0' and saved_count < len(map.hall_same) - 1:
            direction = map.hall_same[saved_count]

        if source == '1' and saved_count < len(map.hall_r2_r1):
            direction = map.hall_r2_r1[saved_count]

        print("Applied direction of walker :", direction)

        # todo, set different rewards and learn
        if saved_count > 0:
            reward = None
            if source == '0':
                max_angle = max(float(direction), float(direction_last))
                min_angle = min(float(direction), float(direction_last))

                diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)

                reward = 1 - diff / 180
                print("single room 's reward is :" + str(reward))
                # td = critic.learn(state_last, reward, state)
                # actor.learn(state_last, action_last, td)

            elif source == '1':
                reward = 1 - map.cal_distance_region(1) / 9
                print("src 1 's reward is :", reward)
                td = critic.learn(state_last, reward, state)
                actor.learn(state_last, action_last, td)

            elif source == '4':
                reward = 1 - map.cal_distance_region(4) / 3
                print("src 4 's reward is :", reward)
                td = critic.learn(state_last, reward, state)
                actor.learn(state_last, action_last, td)

        state_last = state
        direction_last = direction

        # transfer given direction into action index, based on taken direction
        action_last = (direction + 45) % 360 / 45

        print("apply movement ...")

        SSLturning(control, direction)

        # control.speed = STEP_SIZE / FORWARD_SECONDS
        control.radius = 0
        control.omega = 0
        time.sleep(FORWARD_SECONDS)
        control.speed = 0
        print("movement done.")

        map.update_walker_pos(direction)
        saved_count += 1

        # fixme, save online model if reach the source, re-chose actor model path if needed
        if source == "0":
            if 3 <= map.walker_pos_x <= 3.2 and 6.5 <= map.walker_pos_z <= 7.5:
                actor.saver.save(actor.sess, ONLINE_MODEL_PATH)
        elif source == "1":
            if 3.5 <= map.walker_pos_x and map.walker_pos_z >= 6:
                actor.saver.save(actor.sess, ONLINE_MODEL_PATH)


if __name__ == '__main__':
    # cd = Control()
    # loop_record(cd)

    cd = CD.ControlDriver()
    p1 = threading.Thread(target=loop_record, args=(cd,))
    p2 = threading.Thread(target=cd.control_part, args=())
    print("hehe")
    p2.start()
    p1.start()
