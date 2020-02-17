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

TURN_SECONDS = 5
FORWARD_SECONDS = 5
STEP_SIZE = 1

MODEL_PATH = "save/multiple/hole/save100.ckpt"
WAV_PATH = "online_wav/"

"""
    Digital Driver Part
"""


class Control:
    def __init__(self):
        self.omega = 0.1
        self.radius = 0

        # to be determined by distance/time
        self.speed = 0


"""
    2D map, define obstacles, restrict regions, locate walker position.
"""


# corresponding 2D map

class Map:
    def __init__(self):
        # start position
        # mass center of the walker
        self.walker_pos_x = 1.0
        self.walker_pos_z = 1.7

        # world axis indicate walker head
        self.walker_face_to = 0

        # max length of walker, safe distance
        self.walker_length = 1.3

        # determine regions and gates

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
        # if 4.2 < next_z <= 5.7:
        #     if self.walker_length <= next_x:
        #         return True
        #     else:
        #         return False
        #
        # elif 1.7 <= next_z <= 4.2:
        #     if self.walker_length <= next_x <= 3.3 - self.walker_length:
        #         return True
        #     else:
        #         return False
        #
        # elif 0 <= next_z < 1.7 and 0 <= next_x <= 3.3:
        #     return True
        #
        # elif 0 <= next_z < 1.7 and next_x < 0:
        #     if self.walker_length <= next_z <= 1.7 - self.walker_length:
        #         return True
        #     else:
        #         return False
        #
        # elif 0 <= next_z < 1.7 and 3.3 < next_x:
        #     if self.walker_length <= next_z <= 1.7 - self.walker_length:
        #         return True
        #     else:
        #         return False
        #
        # elif next_z < 0:
        #     if next_x <= 1.7 - self.walker_length or next_x >= 1.7 + self.walker_length:
        #         return True
        #     else:
        #         return False
        #
        # else:
        #     print("Out of condition in direction validation ... ")
        x = self.walker_pos_x
        z = self.walker_pos_z

        potential_dirs = [0, 45, 90, 135, 180, 225, 270, 315]

        invalids = []

        if 6.0 < z <= 7.4:
            for dire in potential_dirs:
                if (dire + self.walker_face_to) % 360 in [315, 0, 45]:
                    invalids.append(dire)

            if x < self.walker_length:
                for dire in potential_dirs:
                    if (dire + self.walker_face_to) % 360 in [225, 270, 315]:
                        invalids.append(dire)

            if x >= 3.2:
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


"""
    GCC Processor Part
"""


class GccGenerator:
    def __init__(self):
        self.gcc_width_half = 30

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

    def cal_gcc_online(self, input_dir, save_count):
        for i in range(1, 5):
            mic_name = str(save_count) + "_" + "mic%d" % i + ".wav"
            wav = wave.open(os.path.join(input_dir, mic_name), 'rb')

            n_frame = wav.getnframes()
            fs = wav.getframerate()
            data = np.frombuffer(wav.readframes(n_frame), dtype=np.short)

            locals()['data%d' % i] = data

        gcc_vector = []

        center = int(len(locals()['data%d' % 1]) / 2)

        for i in range(1, 5):
            for j in range(i + 1, 5):
                tau, cc = self.gcc_phat(locals()['data%d' % i], locals()['data%d' % j], fs)
                for k in range(center - self.gcc_width_half, center + self.gcc_width_half + 1):
                    gcc_vector.append(cc[k])

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

        # fixme, when load all variables in, we need reset optimizer
        with tf.variable_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(-self.exp_v)

            self.reset_optimizer = tf.variables_initializer(optimizer.variables())

        self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load_trained_model(self, model_path):
        # fixme, when load models, variables are transmit: layers, adam (not placeholder and op)
        self.saver.restore(self.sess, model_path)
        # load l1, acts_prob and adam vars
        # fixme, after load, init adam
        self.sess.run(self.reset_optimizer)

    # invalid indicates action index
    def output_action(self, s, invalid_actions):
        acts = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        # fixme, mask invalid actions based on invalid actions
        p = acts.ravel()
        p = np.array(p)

        for i in range(self.n_actions):
            if i in invalid_actions:
                p[i] = 0

        # choose invalid action with possible 1
        if p.sum() == 0:
            print("determine invalid action")
            act = np.random.choice(np.arange(acts.shape[1]))
        else:
            p /= p.sum()
            act = np.random.choice(np.arange(acts.shape[1]), p=p)
            # act = np.argmax(p)

        return act, p

    def learn(self, s, a, td):
        # fixme, may modify s
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

        # fixme, global will init actor vars, partly init
        # fixme, need init: layer, optimizer (placeholder and op init is unnecessary)
        # self.sess.run(tf.global_variables_initializer())
        uninitialized_vars = [var for var in tf.global_variables() if 'critic' in var.name or 'Critic' in var.name]

        initialize_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(initialize_op)

    def learn(self, s, r, s_):
        # fixme, need modify s, s_
        # s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s: s, self.v_: v_, self.r: r})
        return td_error


def read_wav(file):
    wav = wave.open(file, 'rb')
    fn = wav.getnframes()  # 获取帧数 207270
    fr = wav.getframerate()  # 获取帧速率 44100
    fw = wav.getsampwidth()  # 获取帧速率 44100
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
    gccGenerator = GccGenerator()
    map = Map()

    actor = Actor(366, 8, lr=0.004)
    critic = Critic(366, 8, lr=0.003, gamma=0.95)

    # todo, fine-tuned pre-train model
    actor.load_trained_model(MODEL_PATH)

    # init at the first step
    state_last = None
    action_last = None
    direction_last = None

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
            wave_output_filename = str(saved_count) + ".wav"
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

        print("producing action ...")

        gcc = gccGenerator.cal_gcc_online(WAV_PATH, saved_count)
        state = np.array(gcc)[np.newaxis, :]

        # todo, define invalids, based on constructed map % restrict regions
        invalids_dire = map.detect_invalid_directions()

        # transform walker direction to mic direction
        invalids_idx = [(i + 45) % 360 / 45 for i in invalids_dire]

        action, _ = actor.output_action(state, invalids_idx)

        # transform mic direction to walker direction
        direction = (action + 6) % 7 * 45

        # bias is 45 degree, ok
        print("Estimated direction is :" + str(direction))

        # todo, set different rewards and learn
        if saved_count > 0:
            max_angle = max(float(direction), float(direction_last))
            min_angle = min(float(direction), float(direction_last))

            diff = min(abs(max_angle - min_angle), 360 - max_angle + min_angle)

            reward = 1 - diff / 180
            print("last step's reward is :" + str(reward))

            # learn
            td = critic.learn(state_last, reward, state)
            actor.learn(state_last, action_last, td)

        state_last = state
        action_last = action
        direction_last = direction

        print("apply movement ...")

        # todo, here for test, readin direction to guide or specific direction
        # direction = int(input())

        # give speed , radius, omega, first turn, then forward
        if direction > 180:
            # turn left
            rad = math.radians(360 - direction)
        else:
            # turn right
            rad = - math.radians(direction)

        control.speed = 0
        control.radius = 0
        control.omega = rad / TURN_SECONDS

        time.sleep(TURN_SECONDS)

        control.speed = - STEP_SIZE / FORWARD_SECONDS
        control.radius = 0
        control.omega = 0

        time.sleep(FORWARD_SECONDS)

        print("movement done.")
        control.speed = 0

        map.update_walker_pos(direction)
        print(map.detect_invalid_directions())
        print(map.detect_which_region())

        # begin next step
        saved_count += 1


# test ok for invalid turning directions
def test_route():
    map = Map()

    print("======")
    print(map.walker_pos_x)
    print(map.walker_pos_z)
    print(map.detect_invalid_directions())

    print("======")
    map.update_walker_pos(180)
    print(map.walker_pos_x)
    print(map.walker_pos_z)
    print(map.detect_invalid_directions())

    print("======")
    map.update_walker_pos(90)
    print(map.walker_pos_x)
    print(map.walker_pos_z)
    print(map.detect_invalid_directions())

    print("======")
    map.update_walker_pos(0)
    print(map.walker_pos_x)
    print(map.walker_pos_z)
    print(map.detect_invalid_directions())

    print("======")
    map.update_walker_pos(0)
    print(map.walker_pos_x)
    print(map.walker_pos_z)
    print(map.detect_invalid_directions())


if __name__ == '__main__':
    cd = Control()
    loop_record(cd)

    # test_route()

    # cd = CD.ControlDriver()
    # p1 = threading.Thread(target=loop_record, args=(cd,))
    # p2 = threading.Thread(target=cd.control_part, args=())
    # print("hehe")
    # p2.start()
    # p1.start()