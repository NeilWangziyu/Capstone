import numpy as np
import pylab as pl
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, filtfilt
import time
from sklearn.svm import SVC
from scipy.signal import find_peaks_cwt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout, Convolution1D,MaxPooling1D
from keras.optimizers import Adam
from keras.models import load_model
import torch
import torch.nn as nn
import torch.utils.data as Data


# -------------------------
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
# ---------------------------
# ##############################################################
# normization
def max_min_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# ##############################################################


EMGDATA = []
EMGLABEL = []
# set up sampling frequency and cutoff frequency
fs = 200
cutoff = 25

time_readstart = time.clock()
print("read start")

rootpath = '/Users/ziyu/PycharmProjects/Mirror/emg_data'

for parent, subdir, filenames in os.walk(rootpath):
  count = 0
  print("reading fold path:",parent)
  for filename in filenames:
    if filename.endswith('.csv') :
      signal_paths = [os.path.join(rootpath,parent, filename)]
      #count for each csv file
      count +=  1
      if signal_paths[0]:
        print(signal_paths[0])
        emg1 = []
        emg2 = []
        emg3 = []
        emg4 = []
        emg5 = []
        emg6 = []
        emg7 = []
        emg8 = []

        with open(signal_paths[0]) as f:
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
        #emg and emg_abs's different



        # use emg4 to cut signal
        emg4_smooth = butter_lowpass_filter(emg4_abs, cutoff, fs)

        sigment_add = []
        cutpoint = []

        #cut off for each gesture
        for index in range(0, len(emg4_smooth), 200):
            # print(index)
            # every 400 points is a segment， overlap is 200 points
            sigment = emg4_smooth[index:(index + 400)]
            sigment_mean = sum(sigment) / len(sigment)
            # print(sigment)

            sigment_add.append(sigment_mean)
        cb = np.array(sigment_add)
        peaks = find_peaks_cwt(-cb, [3])
        #print(peaks)
        for item in peaks:
            cut = item * 200 + 200
            cutpoint.append(cut)

        print('分段点:',cutpoint)

        if len(cutpoint) != 11:
            if len(cutpoint) == 10:
                if cutpoint[0] > 1000:
                    cutpoint.insert(0, 100)
                    #insert point 100 as the first segment
                    print('开始分段点插入后:', cutpoint)
                else:
                    cutpoint.insert(10, len(emg4_smooth) - 100)
                    print('前后分段点插入后:', cutpoint)


            elif len(cutpoint) == 9:
                #both front and back missing
                 cutpoint.insert(0, 100)
                 cutpoint.insert(10, len(emg4_smooth)-100)
                 print('前后分段点插入后:', cutpoint)
            else:
                 print("wrong")

        #cutpoint is the result o segment, make 10 segments

        # i is from 0 - 10
        for i in range(0, len(cutpoint) - 1):
            #print(i)

            gesture_emg1 = np.array(emg1[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg1 = butter_lowpass_filter(gesture_emg1, cutoff, fs)
            x = np.linspace(0, len(gesture_emg1), len(gesture_emg1))
            x_new = np.linspace(0, len(gesture_emg1), 5000)
            tck_emg1 = interpolate.splrep(x, gesture_emg1)
            gesture_emg1_bspline = interpolate.splev(x_new, tck_emg1)
            gesture_emg1_bspline_abs = list(map(abs, gesture_emg1_bspline))


            gesture_emg2 = np.array(emg2[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg2 = butter_lowpass_filter(gesture_emg2, cutoff, fs)
            tck_emg2 = interpolate.splrep(x, gesture_emg2)
            gesture_emg2_bspline = interpolate.splev(x_new, tck_emg2)
            gesture_emg2_bspline_abs = list(map(abs, gesture_emg2_bspline))


            gesture_emg3 = np.array(emg3[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg3 = butter_lowpass_filter(gesture_emg3, cutoff, fs)
            tck_emg3 = interpolate.splrep(x, gesture_emg3)
            gesture_emg3_bspline = interpolate.splev(x_new, tck_emg3)
            gesture_emg3_bspline_abs = list(map(abs, gesture_emg3_bspline))


            gesture_emg4 = np.array(emg4[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg4 = butter_lowpass_filter(gesture_emg4, cutoff, fs)
            tck_emg4 = interpolate.splrep(x, gesture_emg4)
            gesture_emg4_bspline = interpolate.splev(x_new, tck_emg4)
            gesture_emg4_bspline_abs = list(map(abs, gesture_emg4_bspline))


            gesture_emg5 = np.array(emg5[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg5 = butter_lowpass_filter(gesture_emg5, cutoff, fs)
            tck_emg5 = interpolate.splrep(x, gesture_emg5)
            gesture_emg5_bspline = interpolate.splev(x_new, tck_emg5)
            gesture_emg5_bspline_abs = list(map(abs, gesture_emg5_bspline))


            gesture_emg6 = np.array(emg6[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg6 = butter_lowpass_filter(gesture_emg6, cutoff, fs)
            tck_emg6 = interpolate.splrep(x, gesture_emg6)
            gesture_emg6_bspline = interpolate.splev(x_new, tck_emg6)
            gesture_emg6_bspline_abs = list(map(abs, gesture_emg6_bspline))


            gesture_emg7 = np.array(emg7[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg7 = butter_lowpass_filter(gesture_emg7, cutoff, fs)
            tck_emg7 = interpolate.splrep(x, gesture_emg7)
            gesture_emg7_bspline = interpolate.splev(x_new, tck_emg7)
            gesture_emg7_bspline_abs = list(map(abs, gesture_emg7_bspline))


            gesture_emg8 = np.array(emg8[cutpoint[i]:cutpoint[i + 1]])
            gesture_emg8 = butter_lowpass_filter(gesture_emg8, cutoff, fs)
            tck_emg8 = interpolate.splrep(x, gesture_emg8)
            gesture_emg8_bspline = interpolate.splev(x_new, tck_emg8)
            gesture_emg8_bspline_abs = list(map(abs, gesture_emg8_bspline))


            gesture = np.append(gesture_emg1_bspline_abs,gesture_emg2_bspline_abs)
            gesture = np.append(gesture, gesture_emg3_bspline_abs)
            gesture = np.append(gesture, gesture_emg4_bspline_abs)
            gesture = np.append(gesture, gesture_emg5_bspline_abs)
            gesture = np.append(gesture, gesture_emg6_bspline_abs)
            gesture = np.append(gesture, gesture_emg7_bspline_abs)
            gesture = np.append(gesture, gesture_emg8_bspline_abs)

            # gesture = [gesture_emg1_bspline_abs,gesture_emg2_bspline_abs,gesture_emg3_bspline_abs,gesture_emg4_bspline_abs,
            #            gesture_emg5_bspline_abs,gesture_emg6_bspline_abs,gesture_emg7_bspline_abs,gesture_emg8_bspline_abs]

            #print('gesture shape:',len(gesture))

            gesture = max_min_normalization(gesture)
            # plt.plot(gesture)
            # plt.show()
            gesture = gesture.reshape(-1, 8, 5000)
            # plt.plot(gesture[0,:])
            # plt.show()
            # print(gesture.shape)
            # plt.plot(gesture[0,0,:])
            # plt.show()
            EMGDATA.append(gesture)
            EMGLABEL.append(i)





finishREAD = (time.clock() - time_readstart)
print("READING Time used:",finishREAD)
print("totally read csv file number:",count)



print('length of EMGDATA:', len(EMGDATA))
print('length of EMGLABEL',len(EMGLABEL))

np.save("EMGDATA_norm_extended.npy",np.array(EMGDATA))
np.save("EMGLABEL_extended.npy",np.array(EMGLABEL))
