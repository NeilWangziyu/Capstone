import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks_cwt

def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
#
def butter_lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y




def f_cutsignal(siganlpath, fs ,cutoff):
    emg1 = []
    emg2 = []
    emg3 = []
    emg4 = []
    emg5 = []
    emg6 = []
    emg7 = []
    emg8 = []

    with open(siganlpath) as f:
        reader = csv.reader(f)
        # 读取一行，下面的reader中已经没有该行了
        head_row = next(reader)
        for row in reader:
            # 行号从2开始
            # print(reader.line_num, row)
            emg1.append(row[1])
            emg2.append(row[2])
            emg3.append(row[3])
            emg4.append(row[4])
            emg5.append(row[5])
            emg6.append(row[6])
            emg7.append(row[7])
            emg8.append(row[8])

    emg1 = [float(i) for i in emg1 ]
    emg1_abs = list(map(abs, emg1))
    emg2 = [float(i) for i in emg2 ]
    emg2_abs = list(map(abs, emg2))
    emg3 = [float(i) for i in emg3 ]
    emg3_abs = list(map(abs, emg3))
    emg4 = [float(i) for i in emg4 ]
    emg4_abs = list(map(abs, emg4))
    emg5 = [float(i) for i in emg5 ]
    emg5_abs = list(map(abs, emg5))
    emg6 = [float(i) for i in emg6 ]
    emg6_abs = list(map(abs, emg6))
    emg7 = [float(i) for i in emg7 ]
    emg7_abs = list(map(abs, emg7))
    emg8 = [float(i) for i in emg8 ]
    emg8_abs = list(map(abs, emg8))

    plt.plot(emg1, label=u"emg1")
    plt.legend()
    plt.show()
    plt.plot(emg1_abs, label=u"emg1_abs")
    plt.legend()
    plt.show()

    # emg1 - emg8 is the signal in list format

    emg4_smooth = butter_lowpass_filter(emg4_abs, cutoff, fs)

    sigment_add = []

    for i in range(0, len(emg4_smooth), 200):
        #print(i)
        # every 400 points is a segment， overlap is 200 points
        sigment = emg4_smooth[i:(i+400)]
        sigment_mean = sum(sigment)/len(sigment)
        #print(sigment)

        sigment_add.append(sigment_mean)

    #print(sigment_add)
    # print(len(sigment_add))
    # plt.plot(sigment_add)
    # plt.title('means of each sigment')
    # plt.grid()
    # plt.show()
    cb = np.array(sigment_add)
    peaks = find_peaks_cwt(-cb,[3])
    # print(peaks)
    cutpoint = []
    for item in peaks:
        cut = item * 200 + 200
        cutpoint.append(cut)


    return cutpoint


path = '/Users/ziyu/Desktop/Capstone/emg_data/emg_yu_164.csv'
cutpoint = f_cutsignal(path, 200, 25)
print(cutpoint)


