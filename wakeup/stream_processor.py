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

"""
    Pre-settings
"""
CHUNK = 1024
RECORD_DEVICE_NAME = "MacBook Pro 麦克风"
RECORD_WIDTH = 2
CHANNELS = 1
RATE = 16000

FORMAT = pyaudio.paInt16

WAV_PATH = "/Users/xyzhao/Desktop/sound_localization/wakeup/stream_tmp"

# cause there is a bug when filename is the same,
# change the filename every time run the program
now = int(round(time.time()*1000))
RANDOM_PREFIX = time.strftime('%m-%d_%H:%M',time.localtime(now/1000))

def setup_device_index():
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
        if device_name.find(RECORD_DEVICE_NAME) != -1:
            device_index = index
            # break

    if device_index != -1:
        print("find the device")

        print(p.get_device_info_by_index(device_index))
    else:
        print("don't find the device")

    return device_index

device_index = setup_device_index()

def store_frames_to_file(frames, name_id):
    wave_output_filename = RANDOM_PREFIX + "win_%d.wav" % (name_id)
    wf = wave.open(os.path.join(WAV_PATH, wave_output_filename), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(RECORD_WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

"""
    Begin streaming part
"""

# can be large enough
RECORD_SECONDS = 50

# need to contain the word
WINDOW_SECONDS = 1

# number of frames in a stream
frame_num_total = int(RATE / CHUNK * RECORD_SECONDS)

# number of frames in a window
frame_num_win = int(RATE / CHUNK * WINDOW_SECONDS)

# number of frames for one stride
frame_num_stride = 5

# after read how many windows flush the buffer
win_num_flush = 10

# frames buffer from stream, need flush after sometime
frames_buffer = []

# to avoid buffer conflict when do flush
buffer_lock = threading.Lock()

# trigger for flush, init start frame
flush_event = threading.Event()

"""
    Read frame by frame into buffer,
        need time to init and fill in one window size
"""
def read_from_stream():
    global frames_buffer

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index)

    for i in range(0, frame_num_total):
        frame = stream.read(CHUNK)
        frames_buffer.append(frame)

        # if i % frame_num_win == 0 and i != 0:
        #     print("read in a window size")

        # flush the buffer
        if i % (frame_num_win * win_num_flush) == 0 and i != 0:
            print("=====  p1:  set the flush")
            flush_event.set()

            buffer_lock.acquire()
            frames_buffer = []
            buffer_lock.release()
    
    stream.stop_stream()
    stream.close()
    p.terminate()


"""
    use a window contains serveral frames to slide,
        store frames in each window to file.
            use wav file to do inference
"""
def process_from_buffer():
    global frames_buffer

    # init setting
    window_count = 0
    start_frame = 0
    while True:
        frames = []

        if flush_event.is_set() is True:
            print("=====  p2:  detect the flush")
            start_frame = 0
            flush_event.clear()
            time.sleep(WINDOW_SECONDS)
            
        if start_frame >= frame_num_total:
            print("ERROR: start frame out of buffer. ")
            exit()

        buffer_lock.acquire()
        for i in range(0, frame_num_win):
            frames.append(frames_buffer[start_frame + i])
        buffer_lock.release()

        store_frames_to_file(frames, window_count)

        # detect this file
        time.sleep(0.5)

        window_count += 1
        start_frame += frame_num_stride
        # print("process a window")

p1 = threading.Thread(target=read_from_stream, args=())
p2 = threading.Thread(target=process_from_buffer, args=())

p1.start()
time.sleep(2)
time.sleep(WINDOW_SECONDS)

p2.start()

