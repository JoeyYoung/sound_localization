"""
    This is a packet of KWS detection,
        dependent on DNN training part
"""

import pyaudio
import ctypes as ct
import numpy as np
import wave
import math
import matplotlib.pyplot as plt
import pyaudio
import os
import librosa
import librosa.display
import threading
import time
from numpy.linalg import norm
from kws_do_inference import KwsNNet

class KwsDetector:
    def __init__(self, chunk, record_device_name, record_width, channels, rate, format, wav_path):
        self.CHUNK = 1024
        self.RECORD_DEVICE_NAME = "MacBook Pro 麦克风"
        self.RECORD_WIDTH = 2
        self.CHANNELS = 1
        self.RATE = 16000
        self.FORMAT = pyaudio.paInt16
        self.WAV_PATH = "/Users/xyzhao/Desktop/sound_localization/wakeup/stream_tmp"
        self.device_index = self.setup_device_index()

        now = int(round(time.time()*1000))
        self.RANDOM_PREFIX = time.strftime('%m-%d_%H:%M',time.localtime(now/1000))

        """
            init NN model, and load graph
        """
        # self.KwsNet = KwsNNet(os.path.join(self.WAV_PATH, self.RANDOM_PREFIX + "win.wav"), "Pretrained_models/DNN/DNN_M.pb", "Pretrained_models/labels.txt")

        """
            Window settings
        """
        # can be large enough
        self.RECORD_SECONDS = 500

        # need to contain the word
        self.WINDOW_SECONDS = 1

        # number of frames in a stream
        self.frame_num_total = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)

        # number of frames in a window
        self.frame_num_win = int(self.RATE / self.CHUNK * self.WINDOW_SECONDS)

        # number of frames for one stride
        self.frame_num_stride = 3 # 5

        # after read how many windows flush the buffer, large enough since no delay
        self.win_num_flush = 100 # 10

        # frames buffer from stream, need flush after sometime
        self.frames_buffer = []

        # to avoid buffer conflict when do flush
        self.buffer_lock = threading.Lock()

        # trigger for flush, init start frame
        self.flush_event = threading.Event()

        self.end_event = threading.Event()

    def setup_device_index(self):
        device_index = -1
        p = pyaudio.PyAudio()

        """
            Recognize Mic device, before loop
        """
        # scan to get usb device
        for index in range(0, p.get_device_count()):
            info = p.get_device_info_by_index(index)
            device_name = info.get("name")
            print("device_name: ", device_name)

            # find mic usb device
            if device_name.find(self.RECORD_DEVICE_NAME) != -1:
                device_index = index
                # break

        if device_index != -1:
            print("find the device")

            print(p.get_device_info_by_index(device_index))
        else:
            print("don't find the device")

        return device_index

    def store_frames_to_file(self, frames, name_id):
        # set to only one temp wav file in real
        wave_output_filename = self.RANDOM_PREFIX + "win.wav" # % (name_id)
        wf = wave.open(os.path.join(self.WAV_PATH, wave_output_filename), 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.RECORD_WIDTH)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def read_from_stream(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(self.RECORD_WIDTH),
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        input_device_index=self.device_index)

        for i in range(0, self.frame_num_total):
            # if self.end_event.is_set() is True:
            #     break

            frame = stream.read(self.CHUNK)
            self.frames_buffer.append(frame)

            # if i % self.frame_num_win == 0 and i != 0:
            #     print("read in a window size")

            # flush the buffer
            # after a large time duration to avoid high memory useage
            if i % (self.frame_num_win * self.win_num_flush) == 0 and i != 0:
                print("=====  p1:  set the flush")
                self.flush_event.set()

                self.buffer_lock.acquire()
                self.frames_buffer = []
                self.buffer_lock.release()
        
        stream.stop_stream()
        stream.close()
        p.terminate()

    def process_from_buffer(self):
        # KwsNet = KwsNNet(os.path.join(self.WAV_PATH, self.RANDOM_PREFIX + "win.wav"), "Pretrained_models/DNN/DNN_M.pb", "Pretrained_models/labels.txt")
        KwsNet = KwsNNet(os.path.join(self.WAV_PATH, self.RANDOM_PREFIX + "win.wav"), "follow.pb", "tmp/speech_commands_train/follow_labels.txt")
        # init setting
        window_count = 0
        start_frame = 0

        continous_wakeups = 0

        while True:
            frames = []

            if self.flush_event.is_set() is True:
                print("=====  p2:  detect the flush")
                start_frame = 0
                self.flush_event.clear()
                time.sleep(self.WINDOW_SECONDS)
                
            if start_frame >= self.frame_num_total:
                print("ERROR: start frame out of buffer. ")
                exit()

            self.buffer_lock.acquire()
            for i in range(0, self.frame_num_win):
                # detect index out of ranage, wait for p1 to fill the buffer
                while (start_frame + i) >= len(self.frames_buffer):
                    continue
                frames.append(self.frames_buffer[start_frame + i])
            self.buffer_lock.release()

            self.store_frames_to_file(frames, window_count)

            # call DNN part to do inference for this file
            this_frame_status = KwsNet.do_inference()
            if this_frame_status == 1:
                continous_wakeups += 1
                print(continous_wakeups)
            elif this_frame_status == 0:
                continous_wakeups -= 0.3
                if continous_wakeups < 0:
                    continous_wakeups = 0
            # print(continous_wakeups)
            if continous_wakeups >= 3:
                print(" ====== wake up")
                # self.end_event.set()
                # break
            # time.sleep(0.05)

            window_count += 1
            start_frame += self.frame_num_stride
            # print("process a window")

    def slide_win_loop(self):
        p1 = threading.Thread(target=self.read_from_stream, args=())
        p2 = threading.Thread(target=self.process_from_buffer, args=())

        p1.start()
        time.sleep(1)
        time.sleep(self.WINDOW_SECONDS)

        p2.start()

        p1.join()
        p2.join() 


if __name__ == "__main__":
    kws = KwsDetector(1, 2, 3, 4, 5, 6, 7)
    kws.slide_win_loop()