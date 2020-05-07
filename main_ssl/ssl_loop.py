"""
    loop thread to run ssl
"""

from ssl_setup import *
from ssl_gcc_generator import GccGenerator
from ssl_actor_critic import Actor, Critic
from ssl_map import Map
from ssl_audio_processor import *
from ssl_turning import SSLturning
import time
import sys
import os
import threading

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)

import DigitalDriver.ControlandOdometryDriver as CD


class SSL:
    def __init__(self):
        print(" === init SSL part ")

    def loop(self, control, source='test'):
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
    ssl = SSL()
    cd = CD.ControlDriver()
    p1 = threading.Thread(target=ssl.loop, args=(cd,))
    p2 = threading.Thread(target=cd.control_part, args=())

    p2.start()
    p1.start()
