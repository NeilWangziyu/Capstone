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

EMGDATA = []
EMGLABEL = []
# set up sampling frequency and cutoff frequency
fs = 200
cutoff = 25

time_readstart = time.clock()
print("read start")
rootpath = '/Users/ziyu/资料/我的课程与资料/研一下/Capstone/testdata'
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

        #cutpoint is the result of segment, make 10 segments

        # i is from 0 - 10
        for i in range(0, len(cutpoint) - 1):
            #print(i)
            gesture_emg1 = np.array(emg1[cutpoint[i]:cutpoint[i + 1]])
            # plt.plot(gesture)
            #print('length of gesture:%d' % len(gesture_emg1))
            x = np.linspace(0, len(gesture_emg1), len(gesture_emg1))
            #print('length of x = %d'%len(x))
            x_new = np.linspace(0, len(gesture_emg1), 5000)
            # print('length of x_new = %d'%len(x_new))
            tck_emg1 = interpolate.splrep(x, gesture_emg1)
            gesture_emg1_bspline = interpolate.splev(x_new, tck_emg1)
            gesture_emg1_bspline_abs = list(map(abs, gesture_emg1_bspline))
            # print(gesture_bspline)
            # plt.plot(x, gesture, "o", label=u"original data")
            #print('length of gesture after interpolate:%d' % len(gesture_emg1_bspline))
            # plt.plot(gesture_bspline, label=u"B-spline interpolate")
            # pl.legend()
            # pl.show()
            #the gesture_bspline list is the result of one gesture of one emg

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


            # gesture = np.append(gesture_emg1_bspline_abs,gesture_emg2_bspline_abs)
            # gesture = np.append(gesture, gesture_emg3_bspline_abs)
            # gesture = np.append(gesture, gesture_emg4_bspline_abs)
            # gesture = np.append(gesture, gesture_emg5_bspline_abs)
            # gesture = np.append(gesture, gesture_emg6_bspline_abs)
            # gesture = np.append(gesture, gesture_emg7_bspline_abs)
            # gesture = np.append(gesture, gesture_emg8_bspline_abs)
            gesture = [gesture_emg1_bspline_abs,gesture_emg2_bspline_abs,gesture_emg3_bspline_abs,gesture_emg4_bspline_abs,
                       gesture_emg5_bspline_abs,gesture_emg6_bspline_abs,gesture_emg7_bspline_abs,gesture_emg8_bspline_abs]
            #print('gesture shape:',len(gesture))

            EMGDATA.append(gesture)
            EMGLABEL.append(i)



finishREAD = (time.clock() - time_readstart)
print("READING Time used:",finishREAD)
print("totally read csv file number:",count)
# for temp in EMGLABEL:
#     plt.plot(EMGDATA[temp])
#     plt.title('EMGDATA[%d],gesture:%d'%(temp,EMGLABEL[temp]))
#     plt.show()

print('length of EMGDATA:', len(EMGDATA))
print('length of EMGLABEL',len(EMGLABEL))

#print(EMGDATA)
#EMGDATA AND EMGLABEL is the training data


# # PCA transfer
# print('start to PCA')
# startPCA = time.clock()
# # feature wanted
# K=50
# # building model，n_components is the number of feature wanted
# model = pca.PCA(n_components=K).fit(EMGDATA)
# # transform to run PCA
# face_X = model.transform(EMGDATA)
#
# finishPCA = (time.clock() - startPCA)
# print("PCA Time used:",finishPCA)
# print(EMGLABEL)
np.save("EMGDATA_unnorm.npy",np.array(EMGDATA))
np.save("EMGLABEL.npy",np.array(EMGLABEL))
X_train, X_test, y_train, y_test = train_test_split(EMGDATA, EMGLABEL, test_size=0.3)
# X_train = EMGDATA
# y_train = EMGLABEL
print("length of X_train:", len(X_train))
print("feature used", len(X_train[0]))

# ----------------------------------

X_train=np.array(X_train)
X_test=np.array(X_test)
X_train = X_train.reshape(-1, 1, 200, 200)
X_test = X_test.reshape(-1, 1, 200,200)


print('start to train CNN')
startCNN = time.clock()
y_train = np_utils.to_categorical(y_train,num_classes=10)
#print(y_train)
y_test_origin = y_test
y_test = np_utils.to_categorical(y_test,num_classes=10)
print("number of category:10")

model = Sequential()
model.add(Convolution2D(30,(3,3),batch_input_shape=(None,1, 200,200),activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(60,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add((Dropout(0.3)))
model.add(Dense(50,activation='relu'))
model.add((Dropout(0.3)))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
print('Training ------------')

model.fit(X_train, y_train, epochs=20, batch_size=100)

print('model training finished')
model.summary()
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

finishCNN = (time.clock() - startCNN)
print("CNN Time used:",finishCNN)
# print(y_test)
# print(X_result)

#calculate Similiarity
# same = 0
# for num in range(0,len(y_test)-1):
#   if y_test[num] ==  X_result[num]:
#     same = same + 1
# similiarity = same/len(y_test)
# print("similiarity:", same/len(y_test))

#save the architecture of CNN and weights of CNN
model.save('final_CNN_model_no_norm.h5')
print('model is saved as final_CNN_model_no_norm.h5')

#import model and weight


