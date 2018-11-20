import torch
import numpy as np
import time
from scipy.signal import butter, filtfilt
import csv
from scipy import interpolate
import json
import os
import paramiko
from CNN_1D_Training import CNN


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



    x = np.linspace(0, len(emg1_abs), len(emg1_abs))
    x_new = np.linspace(0, len(emg1_abs), 5000)

    gesture_emg1 = np.array(emg1_abs)
    gesture_emg1 = butter_lowpass_filter(gesture_emg1, cutoff, fs)
    tck_emg1 = interpolate.splrep(x, gesture_emg1)
    gesture_emg1_bspline = interpolate.splev(x_new, tck_emg1)
    gesture_emg1_bspline_abs = list(map(abs, gesture_emg1_bspline))


    gesture_emg2 = np.array(emg2_abs)
    gesture_emg2 = butter_lowpass_filter(gesture_emg2, cutoff, fs)
    tck_emg2 = interpolate.splrep(x, gesture_emg2)
    gesture_emg2_bspline = interpolate.splev(x_new, tck_emg2)
    gesture_emg2_bspline_abs = list(map(abs, gesture_emg2_bspline))

    gesture_emg3 = np.array(emg3_abs)
    gesture_emg3 = butter_lowpass_filter(gesture_emg3, cutoff, fs)
    tck_emg3 = interpolate.splrep(x, gesture_emg3)
    gesture_emg3_bspline = interpolate.splev(x_new, tck_emg3)
    gesture_emg3_bspline_abs = list(map(abs, gesture_emg3_bspline))

    gesture_emg4 = np.array(emg4_abs)
    gesture_emg4 = butter_lowpass_filter(gesture_emg4, cutoff, fs)
    tck_emg4 = interpolate.splrep(x, gesture_emg4)
    gesture_emg4_bspline = interpolate.splev(x_new, tck_emg4)
    gesture_emg4_bspline_abs = list(map(abs, gesture_emg4_bspline))

    gesture_emg5 = np.array(emg5_abs)
    gesture_emg5 = butter_lowpass_filter(gesture_emg5, cutoff, fs)
    tck_emg5 = interpolate.splrep(x, gesture_emg5)
    gesture_emg5_bspline = interpolate.splev(x_new, tck_emg5)
    gesture_emg5_bspline_abs = list(map(abs, gesture_emg5_bspline))

    gesture_emg6 = np.array(emg6_abs)
    gesture_emg6 = butter_lowpass_filter(gesture_emg6, cutoff, fs)
    tck_emg6 = interpolate.splrep(x, gesture_emg6)
    gesture_emg6_bspline = interpolate.splev(x_new, tck_emg6)
    gesture_emg6_bspline_abs = list(map(abs, gesture_emg6_bspline))

    gesture_emg7 = np.array(emg7_abs)
    gesture_emg7 = butter_lowpass_filter(gesture_emg7, cutoff, fs)
    tck_emg7 = interpolate.splrep(x, gesture_emg7)
    gesture_emg7_bspline = interpolate.splev(x_new, tck_emg7)
    gesture_emg7_bspline_abs = list(map(abs, gesture_emg7_bspline))

    gesture_emg8 = np.array(emg8_abs)
    gesture_emg8 = butter_lowpass_filter(gesture_emg8, cutoff, fs)
    tck_emg8 = interpolate.splrep(x, gesture_emg8)
    gesture_emg8_bspline = interpolate.splev(x_new, tck_emg8)
    gesture_emg8_bspline_abs = list(map(abs, gesture_emg8_bspline))

    #gesture size：8*5000
    gesture = np.append(gesture_emg1_bspline_abs, gesture_emg2_bspline_abs)
    gesture = np.append(gesture, gesture_emg3_bspline_abs)
    gesture = np.append(gesture, gesture_emg4_bspline_abs)
    gesture = np.append(gesture, gesture_emg5_bspline_abs)
    gesture = np.append(gesture, gesture_emg6_bspline_abs)
    gesture = np.append(gesture, gesture_emg7_bspline_abs)
    gesture = np.append(gesture, gesture_emg8_bspline_abs)


    gesture = max_min_normalization(gesture)

    gesture = gesture.reshape(1, 8, 5000)

    return gesture





if __name__ == "__main__":

    cnn = torch.load('PytorchModel_CNN_1D_norm_extended_all.pkl')
    init_time = "Oct 12 00:35:11"

    while(True):
        start_time = time.clock()
        file_name = "gesture0.csv"
        filemt = time.localtime(os.stat(file_name).st_mtime)
        filemt = time.strftime("%b %d %H:%M:%S", filemt)
        if filemt == init_time:
            print("和上一次时间相同,返回10")
            data = {"name": 10}
            with open("result.json", 'w') as json_file:
                json.dump(data, json_file)

        else:
            init_time = filemt
            emg1 = []
            with open(file_name) as f:
                reader = csv.reader(f)
                # 读取一行，下面的reader中已经没有该行了
                head_row = next(reader)
                for row in reader:
                    # 行号从2开始
                    emg1.append(row[1])
            if len(emg1)<=100 or len(emg1)>=5000:
                print("数据过长，返回10")
                data = {"name": 10}
                with open("result.json", 'w') as json_file:
                    json.dump(data, json_file)


            else:
                gesture = readgesture(file_name)
                print(gesture.shape)
                gesture = torch.from_numpy(gesture).type(torch.FloatTensor)
                test_output = cnn(gesture)[0]
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                print("read",file_name, '这个动作为', pred_y.item())
                data = {"name": pred_y.item()}

                with open("result.json", 'w') as json_file:
                    json.dump(data, json_file)


        #     上传JSON文件
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 跳过了远程连接中选择‘是’的环节,
            ssh.connect('119.23.243.57', 22, 'root', 'Fdan123@')
            transport = paramiko.Transport(('119.23.243.57', 22))
            transport.connect(username='root', password='Fdan123@')
            stdin, stdout, stderr = ssh.exec_command('df')  # ssh 协议栈命令
            # for line in stdout:  # 逐行打印回显
            #     print(line)
            sftp = paramiko.SFTPClient.from_transport(transport)
            # 将location.py 上传至服务器 /tmp/test.py,都是绝对路径，不是文件夹
            sftp.put('result.json', 'result/result.json')
            transport.close()
            elapsed = (time.clock() - start_time)
            print("完成传输，用时：",elapsed)

        time.sleep(1)






