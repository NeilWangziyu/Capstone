import numpy as np
import pylab as pl
from scipy import interpolate
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import csv
from keras.models import load_model

fs = 200
cutoff = 25

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
# ##############################################################
# normization
def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# ##############################################################



def readgesture(file):
    emg1 = []
    emg2 = []
    emg3 = []
    emg4 = []
    emg5 = []
    emg6 = []
    emg7 = []
    emg8 = []
    with open(file) as f:
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

    emg1 = [float(i) for i in emg1]
    emg1_abs = list(map(abs, emg1))
    emg2 = [float(i) for i in emg2]
    emg2_abs = list(map(abs, emg2))
    emg3 = [float(i) for i in emg3]
    emg3_abs = list(map(abs, emg3))
    emg4 = [float(i) for i in emg4]
    emg4_abs = list(map(abs, emg4))
    emg5 = [float(i) for i in emg5]
    emg5_abs = list(map(abs, emg5))
    emg6 = [float(i) for i in emg6]
    emg6_abs = list(map(abs, emg6))
    emg7 = [float(i) for i in emg7]
    emg7_abs = list(map(abs, emg7))
    emg8 = [float(i) for i in emg8]
    emg8_abs = list(map(abs, emg8))
    # emg and emg_abs's different

    # plt.plot(emg1_abs)
    # plt.show()

    x = np.linspace(0, len(emg1_abs), len(emg1_abs))
    x_new = np.linspace(0, len(emg1_abs), 5000)

    gesture_emg1 = np.array(emg1_abs)
    tck_emg1 = interpolate.splrep(x, gesture_emg1)
    gesture_emg1_bspline = interpolate.splev(x_new, tck_emg1)
    # plt.plot(gesture_emg1_bspline)
    # plt.show()
    gesture_emg1_bspline_abs = list(map(abs, gesture_emg1_bspline))
    # plt.plot(gesture_emg1_bspline_abs)
    # plt.show()

    gesture_emg2 = np.array(emg2_abs)
    tck_emg2 = interpolate.splrep(x, gesture_emg2)
    gesture_emg2_bspline = interpolate.splev(x_new, tck_emg2)
    gesture_emg2_bspline_abs = list(map(abs, gesture_emg2_bspline))

    gesture_emg3 = np.array(emg3_abs)
    tck_emg3 = interpolate.splrep(x, gesture_emg3)
    gesture_emg3_bspline = interpolate.splev(x_new, tck_emg3)
    gesture_emg3_bspline_abs = list(map(abs, gesture_emg3_bspline))

    gesture_emg4 = np.array(emg4_abs)
    tck_emg4 = interpolate.splrep(x, gesture_emg4)
    gesture_emg4_bspline = interpolate.splev(x_new, tck_emg4)
    gesture_emg4_bspline_abs = list(map(abs, gesture_emg4_bspline))

    gesture_emg5 = np.array(emg5_abs)
    tck_emg5 = interpolate.splrep(x, gesture_emg5)
    gesture_emg5_bspline = interpolate.splev(x_new, tck_emg5)
    gesture_emg5_bspline_abs = list(map(abs, gesture_emg5_bspline))

    gesture_emg5 = np.array(emg5_abs)
    tck_emg5 = interpolate.splrep(x, gesture_emg5)
    gesture_emg5_bspline = interpolate.splev(x_new, tck_emg5)
    gesture_emg5_bspline_abs = list(map(abs, gesture_emg5_bspline))

    gesture_emg6 = np.array(emg6_abs)
    tck_emg6 = interpolate.splrep(x, gesture_emg6)
    gesture_emg6_bspline = interpolate.splev(x_new, tck_emg6)
    gesture_emg6_bspline_abs = list(map(abs, gesture_emg6_bspline))

    gesture_emg7 = np.array(emg7_abs)
    tck_emg7 = interpolate.splrep(x, gesture_emg7)
    gesture_emg7_bspline = interpolate.splev(x_new, tck_emg7)
    gesture_emg7_bspline_abs = list(map(abs, gesture_emg7_bspline))

    gesture_emg8 = np.array(emg8_abs)
    tck_emg8 = interpolate.splrep(x, gesture_emg8)
    gesture_emg8_bspline = interpolate.splev(x_new, tck_emg8)
    gesture_emg8_bspline_abs = list(map(abs, gesture_emg8_bspline))

    # normalization
    gesture_emg1_bspline_abs = max_min_normalization(gesture_emg1_bspline_abs)
    gesture_emg2_bspline_abs = max_min_normalization(gesture_emg2_bspline_abs)
    gesture_emg3_bspline_abs = max_min_normalization(gesture_emg3_bspline_abs)
    gesture_emg4_bspline_abs = max_min_normalization(gesture_emg4_bspline_abs)
    gesture_emg5_bspline_abs = max_min_normalization(gesture_emg5_bspline_abs)
    gesture_emg6_bspline_abs = max_min_normalization(gesture_emg6_bspline_abs)
    gesture_emg7_bspline_abs = max_min_normalization(gesture_emg7_bspline_abs)
    gesture_emg8_bspline_abs = max_min_normalization(gesture_emg8_bspline_abs)

    #gesture size：5000*8
    gesture = np.append(gesture_emg1_bspline_abs,gesture_emg2_bspline_abs)
    gesture = np.append(gesture, gesture_emg3_bspline_abs)
    gesture = np.append(gesture, gesture_emg4_bspline_abs)
    gesture = np.append(gesture, gesture_emg5_bspline_abs)
    gesture = np.append(gesture, gesture_emg6_bspline_abs)
    gesture = np.append(gesture, gesture_emg7_bspline_abs)
    gesture = np.append(gesture, gesture_emg8_bspline_abs)

    # smoothed
    gesture = butter_lowpass_filter(gesture, cutoff, fs)

    # reshape gesture
    gesture = np.array(gesture)
    gesture = gesture.reshape(-1, 1, 200, 200)
    return gesture


if __name__ == '__main__':
    # load my model
    model = load_model('my_CNN_model.h5')

    file = 'one_gesture.csv'
    gesture = readgesture(file)
    result = model.predict(gesture)
    result = result.argmax(axis=-1)

    print('这个动作为：',result)
