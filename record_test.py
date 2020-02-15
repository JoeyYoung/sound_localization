# Sound Source locate
# 
# @Time    : 2019-11-01 22:12
# @Author  : xyzhao
# @File    : record_test.py
# @Description:

import pyaudio
import wave
from scipy.io import wavfile
import sys
import numpy as np

CHUNK = 1024
RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
RECORD_WIDTH = 2
CHANNELS = 4
RATE = 16000
RECORD_SECONDS = 5


"""
    python record_test.py file
"""
if __name__ == '__main__':
    device_index = -1

    p = pyaudio.PyAudio()

    """
        Recognize Mic device
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

    """
        Record
    """
    WAVE_OUTPUT_FILENAME = sys.argv[1]

    # format: sample size
    stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index)
    print("start....")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("done...")

    """
        Write into one wave file 
    """

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(RECORD_WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    """
        Split into four channels
    """
    sampleRate, musicData = wavfile.read(WAVE_OUTPUT_FILENAME)

    mic1 = []
    mic2 = []
    mic3 = []
    mic4 = []
    for item in musicData:
        mic1.append(item[0])
        mic2.append(item[1])
        mic3.append(item[2])
        mic4.append(item[3])

    front = WAVE_OUTPUT_FILENAME[:len(WAVE_OUTPUT_FILENAME) - 4]
    # physic mic number --- channel number
    wavfile.write(front + '_mic1.wav', sampleRate, np.array(mic1))
    wavfile.write(front + '_mic2.wav', sampleRate, np.array(mic4))
    wavfile.write(front + '_mic3.wav', sampleRate, np.array(mic2))
    wavfile.write(front + '_mic4.wav', sampleRate, np.array(mic3))
