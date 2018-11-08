import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import KFold
import torch
import numpy as np
import time
from scipy.signal import butter, filtfilt
import csv
from scipy import interpolate

fs = 200
cutoff = 25

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=500,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(128, 32)
        self.drop = nn.Dropout(p=0.4)
        self.output = nn.Linear(32, 10)
        self.logsoftmax = nn.LogSoftmax()



    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        x = self.out(r_out[:, -1, :])  # 选取最后一个时间点的output（看完整段信号之后进行判断）
        x = self.drop(x)
        x = self.output(x)
        output = self.logsoftmax(x)
        return output


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
    x_new = np.linspace(0, len(emg1_abs), 500)

    gesture_emg1 = np.array(emg1_abs)
    gesture_emg1 = butter_lowpass_filter(gesture_emg1, cutoff, fs)
    tck_emg1 = interpolate.splrep(x, gesture_emg1)
    gesture_emg1_bspline = interpolate.splev(x_new, tck_emg1)
    # plt.plot(gesture_emg1_bspline)
    # plt.show()
    gesture_emg1_bspline_abs = list(map(abs, gesture_emg1_bspline))
    # plt.plot(gesture_emg1_bspline_abs)
    # plt.show()

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

    #gesture size：8*500
    gesture = np.append(gesture_emg1_bspline_abs, gesture_emg2_bspline_abs)
    gesture = np.append(gesture, gesture_emg3_bspline_abs)
    gesture = np.append(gesture, gesture_emg4_bspline_abs)
    gesture = np.append(gesture, gesture_emg5_bspline_abs)
    gesture = np.append(gesture, gesture_emg6_bspline_abs)
    gesture = np.append(gesture, gesture_emg7_bspline_abs)
    gesture = np.append(gesture, gesture_emg8_bspline_abs)
    gesture = max_min_normalization(gesture)
    gesture = gesture.reshape(1, 8, 500)
    return gesture



if __name__ == "__main__":

    time_readstart = time.clock()

    EMGDATA = np.load("EMGDATA_norm_extended_desampled.npy")
    EMGLABEL = np.load("EMGLABEL_extended_desampled.npy")

    finishREAD = (time.clock() - time_readstart)

    print("READING Time used:", finishREAD)
    print('length of EMGDATA:', len(EMGDATA))
    print('length of EMGLABEL', len(EMGLABEL))

    EMGDATA = np.array(EMGDATA)
    EMGLABEL = np.array(EMGLABEL)

    result = []
    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(EMGDATA):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = EMGDATA[train_index], EMGDATA[test_index]
        y_train, y_test = EMGLABEL[train_index], EMGLABEL[test_index]

        train_data = torch.from_numpy(np.array(X_train)).type(torch.FloatTensor)
        test_data = torch.from_numpy(np.array(X_test)).type(torch.FloatTensor)
        train_label = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
        test_label = torch.from_numpy(np.array(y_test)).type(torch.LongTensor)

        print(train_data.shape)

        train_Dataset = Data.TensorDataset(train_data, train_label)
        train_loader = Data.DataLoader(
            dataset=train_Dataset,
            batch_size=64,
            shuffle=True)

        test_Dataset = Data.TensorDataset(test_data, test_label)
        test_loader = Data.DataLoader(
            dataset=test_Dataset,
            batch_size=64,
            shuffle=True)

        rnn = RNN()
        print(rnn)

        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(40):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.view(-1, 8, 500)
                # print(b_x.shape)
                output = rnn(b_x)  # rnn output
                loss = loss_func(output, b_y)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

            if epoch % 10 == 0:
                correct = 0
                total = 0
                for step, (t_x, t_y) in enumerate(
                        test_loader):  # gives batch data, normalize x when iterate train_loader
                    t_x = t_x.view(-1, 8, 500)
                    test_output = rnn(t_x)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    right = torch.sum(pred_y == t_y).type(torch.FloatTensor)
                    total += t_y.size()[0]
                    correct += right
                accuracy = correct / total
                print("训练", epoch, "后，测试", total, "个数据, ", "准确率为：", accuracy.data)

        print("训练完成，开始测试：")
        correct = 0
        total = 0
        for step, (t_x, t_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
            t_x = t_x.view(-1, 8, 500)
            test_output = rnn(t_x)
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # print("pred:", pred_y)
            # print("y", t_y)
            right = torch.sum(pred_y == t_y).type(torch.FloatTensor)
            total += t_y.size()[0]
            correct += right
        accuracy = correct / total
        print("训练", epoch + 1, "次后，测试", total, "个数据, ", "准确率为：", accuracy.data)
        result.append(accuracy.data)

    #     print("本次训练准确率:", score_this_fold)
    print("10折平均准确率：", np.mean(result))



    # 采用所有信号进行训练:
    #
    final_train_data = torch.from_numpy(np.array(EMGDATA)).type(torch.FloatTensor)
    final_train_label = torch.from_numpy(np.array(EMGLABEL)).type(torch.LongTensor)

    final_train_Dataset = Data.TensorDataset(train_data, train_label)
    final_train_loader = Data.DataLoader(
        dataset=final_train_Dataset,
        batch_size=64,
        shuffle=True)

    final_rnn = RNN()
    print(final_rnn)

    optimizer_final = torch.optim.Adam(final_rnn.parameters(), lr=0.01)
    loss_func_final = nn.CrossEntropyLoss()

    for epoch in range(40):
        for step, (b_x, b_y) in enumerate(final_train_loader):
            b_x = b_x.view(-1, 8, 500)
            # print(b_x.shape)
            output = final_rnn(b_x)  # rnn output
            loss = loss_func_final(output, b_y)  # cross entropy loss
            optimizer_final.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer_final.step()


    print("采用此模型对信号进行测试：")
    file = 'gesture1.csv'
    gesture = readgesture(file)
    gesture = torch.from_numpy(gesture).type(torch.FloatTensor)
    gesture = gesture.view(-1, 8, 500)
    test_output = final_rnn(gesture)
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    print("pred:", pred_y)
    print("true label:", 1)


    print("采用此模型对信号进行测试：")
    file = 'gesture0.csv'
    gesture = readgesture(file)
    gesture = torch.from_numpy(gesture).type(torch.FloatTensor)
    gesture = gesture.view(-1, 8, 500)
    test_output = final_rnn(gesture)
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    print("pred:", pred_y)
    print("true label:", 0)

    # torch.save(final_rnn, "PytorchModel_RNN_all_extended.pkl")

    for i in range(10):
        time_start = time.clock()
        file = 'Test_10/{}.csv'.format(i)
        # file = 'gesture1.csv'
        gesture = readgesture(file)
        gesture = torch.from_numpy(gesture).type(torch.FloatTensor)
        gesture = gesture.view(-1, 8, 500)
        test_output = final_rnn(gesture)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        elapsed = (time.clock() - time_start)
        print(i,'这个动作为',pred_y.item(),' 使用时间：',elapsed)
    print("\n")

    for i in range(10):
        time_start = time.clock()
        file = 'figure2-10.12/{}.csv'.format(i)
        # file = 'gesture1.csv'
        gesture = readgesture(file)
        gesture = torch.from_numpy(gesture).type(torch.FloatTensor)
        gesture = gesture.view(-1, 8, 500)
        test_output = final_rnn(gesture)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        elapsed = (time.clock() - time_start)
        print(i,'这个动作为',pred_y.item(),' 使用时间：',elapsed)

