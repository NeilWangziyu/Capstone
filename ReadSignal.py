import csv
import pylab as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks_cwt
from scipy import interpolate


# filter Butterworth

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








emg1 = []
emg2 = []
emg3 = []
emg4 = []
emg5 = []
emg6 = []
emg7 = []
emg8 = []

filename = '/Users/ziyu/Desktop/Capstone/emg_data/emg_yu_169.csv'
with open(filename) as f:
    reader = csv.reader(f)
    # 读取一行，下面的reader中已经没有该行了
    head_row = next(reader)
    for row in reader:
        # 行号从2开始
        #print(reader.line_num, row)
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
# emg1 - emg8 is the signal


# print(emg1)
# print(len(emg1))
#
# plt.subplot(241)
# plt.plot(emg1)
# plt.title('emg1')
# plt.subplot(242)
# plt.plot(emg2)
# plt.title('emg2')
#
# plt.subplot(243)
# plt.plot(emg3)
# plt.title('emg3')
#
# plt.subplot(244)
# plt.plot(emg4)
# plt.title('emg4')
#
# plt.subplot(245)
# plt.plot(emg5)
# plt.title('emg5')
#
# plt.subplot(246)
# plt.plot(emg6)
# plt.title('emg6')
#
# plt.subplot(247)
# plt.plot(emg7)
# plt.title('emg7')
#
# plt.subplot(248)
# plt.plot(emg8)
# plt.title('emg8')
#
# plt.show()

#
fs = 200
cutoff = 25
# emg2_smooth = butter_lowpass_filter(emg2, cutoff, fs)
# plt.plot(emg2_smooth)

#use emg4 to cut signal
emg4_smooth = butter_lowpass_filter(emg4_abs, cutoff, fs)
plt.plot(emg4_smooth)
plt.title('emg4_smooth')
plt.show()

plt.plot(emg4)
plt.title('emg4')
plt.show()

sigment_add = []

for i in range(0, len(emg4_smooth), 200):
    #print(i)
    # every 400 points is a segment， overlap is 200 points
    sigment = emg4_smooth[i:(i+400)]
    sigment_mean = sum(sigment)/len(sigment)
    #print(sigment)

    sigment_add.append(sigment_mean)

# print(sigment_add)
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

print(cutpoint)
#print(len(emg4))
for i in range(0,len(cutpoint)-1):
    #print('i=',i)
    #plt.subplot(2, 5, i+1)
    #plt.title('gesture%d' % (i+1))
    gesture = np.array(emg4[cutpoint[i]:cutpoint[i+1]])
    #plt.plot(gesture)
    print('length of gesture:%d' % len(gesture))
    x = np.linspace(0,len(gesture),len(gesture))
    #print('length of x = %d'%len(x))
    x_new = np.linspace(0, len(gesture), 5000)
    #print('length of x_new = %d'%len(x_new))
    tck = interpolate.splrep(x, gesture)
    gesture_bspline = interpolate.splev(x_new, tck)
    #print(gesture_bspline)
    #plt.plot(x, gesture, "o", label=u"original data")
    print('length of gesture after interpolate:%d' % len(gesture_bspline))
    plt.plot(gesture_bspline, label=u"B-spline interpolate")
    plt.legend()
    plt.show()

#plt.show()



#
# plt.subplot(2, 5, 1)
# plt.plot(gesture[1])
# plt.title('gesture1')
# plt.subplot(2, 5, 2)
# plt.plot(gesture[2])
# plt.title('gesture2')
#
# plt.subplot(2, 5, 3)
# plt.plot(gesture[3])
# plt.title('gesture3')
#
# plt.subplot(2, 5, 4)
# plt.plot(gesture[4])
# plt.title('gesture4')
#
# plt.subplot(2, 5, 5)
# plt.plot(gesture[5])
# plt.title('gesture5')
#
# plt.subplot(2, 5, 6)
# plt.plot(gesture[6])
# plt.title('gesture6')
#
# plt.subplot(2, 5, 7)
# plt.plot(gesture[7])
# plt.title('gesture7')
#
# plt.subplot(2, 5, 8)
# plt.plot(gesture[8])
# plt.title('gesture8')
#
# plt.subplot(2, 5, 9)
# plt.plot(gesture[9])
# plt.title('gesture10')
#
# plt.subplot(2, 5, 10)
# plt.plot(gesture[10])
# plt.title('gesture10')
#
# plt.show()



# plt.grid()
# plt.show()
#
# cutemg=function(x) {
#
# n=1
#
# wd=c()
#
# for (i in 1:(nrow(x)/25-1)) {
#
# wd=cbind(wd,apply(x[,-1],2,FUN=function(y) mean(abs(y[n:(n+50)]))))
#
# n=n+25}
#

# wdm=apply(wd,2,mean)
#
# wdrank=order(wdm,decreasing=F)
#
# mins=wdrank[1:300][order(wdrank[1:300])]
#
# mindf=c()
#
# for (i in 1:(length(mins)-1)) {
#
# mindf=c(mindf,mins[i+1]-mins[i])
#
# }
#
# pos=order(mindf,decreasing=T)[1:10]
#
# poso=pos[order(pos,decreasing=F)]
#
# posw=c()
#
# for (i in 1:(length(poso)-1)) {
#
# posw=c(posw,floor(mean(c(poso[i],poso[i+1]))))}
#
# cowd=mins[posw]
#
# cutpos=cowd*25-12
#
# gesdf=list(ges1=x[1:cutpos[1],-1],ges2=x[(cutpos[1]+1):cutpos[2],-1],ges3=x[(cutpos[2]+1):cutpos[3],-1],ges4=x[(cutpos[3]+1):cutpos[4],-1],ges5=x[(cutpos[4]+1):cutpos[5],-1],ges6=x[(cutpos[5]+1):cutpos[6],-1],ges7=x[(cutpos[6]+1):cutpos[7],-1],ges8=x[(cutpos[7]+1):cutpos[8],-1],ges9=x[(cutpos[8]+1):cutpos[9],-1],ges10=x[(cutpos[9]+1):nrow(x),-1])
#
# return(gesdf)
#
# }