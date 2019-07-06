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


def read_data(rootpath):
    EMGDATA = []
    EMGLABEL = []
    # set up sampling frequency and cutoff frequency
    fs = 200
    cutoff = 25

    for parent, subdir, filenames in os.walk(rootpath):
        count = 0
        # print("reading fold path:", parent)
        for filename in filenames:
            if filename.endswith('.csv'):
                signal_paths = [os.path.join(rootpath, parent, filename)]
                # count for each csv file
                count += 1
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
                    # emg and emg_abs's different

                    # use emg4 to cut signal
                    emg4_smooth = butter_lowpass_filter(emg4_abs, cutoff, fs)

                    sigment_add = []
                    cutpoint = []

                    # cut off for each gesture
                    for index in range(0, len(emg4_smooth), 200):
                        # print(index)
                        # every 400 points is a segment， overlap is 200 points
                        sigment = emg4_smooth[index:(index + 400)]
                        sigment_mean = sum(sigment) / len(sigment)
                        # print(sigment)

                        sigment_add.append(sigment_mean)
                    cb = np.array(sigment_add)
                    peaks = find_peaks_cwt(-cb, [3])
                    # print(peaks)
                    for item in peaks:
                        cut = item * 200 + 200
                        cutpoint.append(cut)

                    # print('分段点:', cutpoint)

                    if len(cutpoint) != 11:
                        if len(cutpoint) == 10:
                            if cutpoint[0] > 1000:
                                cutpoint.insert(0, 100)
                                # insert point 100 as the first segment
                                # print('开始分段点插入后:', cutpoint)
                            else:
                                cutpoint.insert(10, len(emg4_smooth) - 100)
                                # print('前后分段点插入后:', cutpoint)


                        elif len(cutpoint) == 9:
                            # both front and back missing
                            cutpoint.insert(0, 100)
                            cutpoint.insert(10, len(emg4_smooth) - 100)
                            # print('前后分段点插入后:', cutpoint)
                        else:
                            print("wrong")

                    # cutpoint is the result of segment, make 10 segments

                    # i is from 0 - 10
                    for i in range(0, len(cutpoint) - 1):
                        # print(i)
                        gesture_emg1 = np.array(emg1[cutpoint[i]:cutpoint[i + 1]])
                        # plt.plot(gesture)
                        # print('length of gesture:%d' % len(gesture_emg1))
                        x = np.linspace(0, len(gesture_emg1), len(gesture_emg1))
                        # print('length of x = %d'%len(x))
                        x_new = np.linspace(0, len(gesture_emg1), 5000)
                        # print('length of x_new = %d'%len(x_new))
                        tck_emg1 = interpolate.splrep(x, gesture_emg1)
                        gesture_emg1_bspline = interpolate.splev(x_new, tck_emg1)
                        gesture_emg1_bspline_abs = list(map(abs, gesture_emg1_bspline))
                        # print(gesture_bspline)
                        # plt.plot(x, gesture, "o", label=u"original data")
                        # print('length of gesture after interpolate:%d' % len(gesture_emg1_bspline))
                        # plt.plot(gesture_bspline, label=u"B-spline interpolate")
                        # pl.legend()
                        # pl.show()
                        # the gesture_bspline list is the result of one gesture of one emg

                        gesture_emg2 = np.array(emg2[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg2 = interpolate.splrep(x, gesture_emg2)
                        gesture_emg2_bspline = interpolate.splev(x_new, tck_emg2)
                        gesture_emg2_bspline_abs = list(map(abs, gesture_emg2_bspline))

                        gesture_emg3 = np.array(emg3[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg3 = interpolate.splrep(x, gesture_emg3)
                        gesture_emg3_bspline = interpolate.splev(x_new, tck_emg3)
                        gesture_emg3_bspline_abs = list(map(abs, gesture_emg3_bspline))

                        gesture_emg4 = np.array(emg4[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg4 = interpolate.splrep(x, gesture_emg4)
                        gesture_emg4_bspline = interpolate.splev(x_new, tck_emg4)
                        gesture_emg4_bspline_abs = list(map(abs, gesture_emg4_bspline))

                        gesture_emg5 = np.array(emg5[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg5 = interpolate.splrep(x, gesture_emg5)
                        gesture_emg5_bspline = interpolate.splev(x_new, tck_emg5)
                        gesture_emg5_bspline_abs = list(map(abs, gesture_emg5_bspline))

                        gesture_emg6 = np.array(emg6[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg6 = interpolate.splrep(x, gesture_emg6)
                        gesture_emg6_bspline = interpolate.splev(x_new, tck_emg6)
                        gesture_emg6_bspline_abs = list(map(abs, gesture_emg6_bspline))

                        gesture_emg7 = np.array(emg7[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg7 = interpolate.splrep(x, gesture_emg7)
                        gesture_emg7_bspline = interpolate.splev(x_new, tck_emg7)
                        gesture_emg7_bspline_abs = list(map(abs, gesture_emg7_bspline))

                        gesture_emg8 = np.array(emg8[cutpoint[i]:cutpoint[i + 1]])
                        tck_emg8 = interpolate.splrep(x, gesture_emg8)
                        gesture_emg8_bspline = interpolate.splev(x_new, tck_emg8)
                        gesture_emg8_bspline_abs = list(map(abs, gesture_emg8_bspline))

                        gesture = np.append(gesture_emg1_bspline_abs, gesture_emg2_bspline_abs)
                        gesture = np.append(gesture, gesture_emg3_bspline_abs)
                        gesture = np.append(gesture, gesture_emg4_bspline_abs)
                        gesture = np.append(gesture, gesture_emg5_bspline_abs)
                        gesture = np.append(gesture, gesture_emg6_bspline_abs)
                        gesture = np.append(gesture, gesture_emg7_bspline_abs)
                        gesture = np.append(gesture, gesture_emg8_bspline_abs)

                        # print('gesture shape:',len(gesture))

                        EMGDATA.append(max_min_normalization(gesture))
                        EMGLABEL.append(i)

    print("totally read csv file number:", count)
    return EMGDATA, EMGLABEL



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 30,
                kernel_size = (3,3),
                stride=1,
                padding=2
                # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            ),# output shape (16, 241, 17)

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,2)),
            # output shape (16, 121, 9)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(30, 15, (3,3), 1, 2),  # (32, 121, 9)
            nn.ReLU(),  # (32, 121, 9))
            nn.MaxPool2d(kernel_size=(2,2)),# (32, 61, 5))
        )

        self.dense1 = nn.Linear(39015, 128)
        self.drop1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(128, 50)
        self.drop2 = nn.Dropout(p=0.2)
        self.out = nn.Linear(50, 10)
        self.logsoftmax = nn.LogSoftmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.out(x)
        output = self.logsoftmax(x)
        return output, x




if __name__ == "__main__":
    rootpath = '/Users/ziyu/资料/我的课程与资料/研一下/Capstone/testdata'
    print("start to read under dir:", rootpath)
    time_readstart = time.clock()

    EMGDATA, EMGLABEL = read_data(rootpath)

    finishREAD = (time.clock() - time_readstart)
    print("READING Time used:", finishREAD)
    print('length of EMGDATA:', len(EMGDATA))
    print('length of EMGLABEL', len(EMGLABEL))

    # plt.plot(EMGDATA[0])
    # plt.title(EMGLABEL[0])
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(EMGDATA, EMGLABEL, test_size=0.3)
    print("length of X_train:", len(X_train))
    print("feature used", len(X_train[0]))

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = torch.from_numpy(np.array(X_train)).type(torch.FloatTensor)
    X_test = torch.from_numpy(np.array(X_test)).type(torch.FloatTensor)
    y_train = torch.from_numpy(np.array(y_train))
    y_test = torch.from_numpy(np.array(y_test))

    # ----------------------------------

    X_train = X_train.reshape(-1, 1, 200, 200)
    X_test = X_test.reshape(-1, 1, 200, 200)


    train_Dataset = Data.TensorDataset(X_train, y_train)

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(
        dataset=train_Dataset,
        batch_size=100,
        shuffle=True)

    cnn = CNN()
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    for epoch in range(20):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

            output = cnn(b_x)[0]                          # cnn output
            loss = loss_func(output, b_y)               # cross entropy loss
            optimizer.zero_grad()                       # clear gradients for this training step
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                test_output, last_layer = cnn(X_test)
                pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
                accuracy = float((pred_y == y_test.data.numpy()).astype(int).sum()) / float(y_test.size(0))
                print('Epoch: ', epoch, '| train loss: %.6f' %loss.data.numpy(), ' | test accuracy: %.3f' % accuracy)




