# Sound Source locate
# 
# @Time    : 2019-10-18 19:31
# @Author  : xyzhao
# @File    : frequency.py
# @Description:  frequency analyze

import wave
import pyaudio
import numpy
import pylab
import numpy as np

if __name__ == '__main__':

    wf = wave.open("./wavdata/8x3x8_src_-2_1.6_-2/walker_2.0_1_2.0_315_mic1.wav", "rb")

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    nframes = wf.getnframes()
    framerate = wf.getframerate()
    # 读取完整的帧数据到str_data中，这是一个string类型的数据
    str_data = wf.readframes(nframes)
    wf.close()
    # 将波形数据转换为数组
    # A new 1-D array initialized from raw binary or text data in a string.
    wave_data = numpy.fromstring(str_data, dtype=numpy.short)
    if len(wave_data) % 2 != 0:
        wave_data = np.append(wave_data, 0)
    # 将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
    wave_data.shape = -1, 2
    # 将数组转置
    wave_data = wave_data.T
    # time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
    time = [i for i in range(len(wave_data[0]))]
    # numpy.arange(0, nframes) * (1.0 / framerate)
    # 绘制波形图
    pylab.plot(time, wave_data[0])
    pylab.subplot(212)
    pylab.plot(time, wave_data[1], c="g")
    pylab.xlabel("time (seconds)")
    pylab.show()
    #
    # 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
    # 采样点数N = 44100
    # start = 0  # 开始采样位置
    # df = framerate / (N - 1)  # 分辨率
    # freq = [df * n for n in range(0, N)]  # N个元素
    # wave_data2 = wave_data[0][start:start + N]
    # c = numpy.fft.fft(wave_data2) * 2 / N
    # # 常规显示采样频率一半的频谱
    # d = int(len(c) / 2)
    # print(d)
    # # 仅显示频率在4000以下的频谱
    # while freq[d] > 300:
    #     d -= 10
    # print(d)
    # pylab.plot(freq[:d - 1], abs(c[:d - 1]), 'r')
    # pylab.show()
