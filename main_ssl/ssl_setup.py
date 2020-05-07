import pyaudio

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
