import numpy as np
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.utils.data as Data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels = 8,
                out_channels = 30,
                kernel_size = 15,
                stride=1,
                padding=2
            ),

            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=30,
                out_channels=60,
                kernel_size=3,
                stride=1,
                padding=2),  # (32, 121, 9)
            nn.ReLU(),  # (32, 121, 9))
            nn.MaxPool1d(kernel_size=2),# (32, 61, 5))
        )

        self.dense1 = nn.Linear(74880, 128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(128, 50)
        self.drop2 = nn.Dropout(p=0.3)
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

    time_readstart = time.clock()

    EMGDATA = np.load("EMGDATA_norm.npy")
    EMGLABEL = np.load("EMGLABEL.npy")

    finishREAD = (time.clock() - time_readstart)

    print("READING Time used:", finishREAD)
    print('length of EMGDATA:', len(EMGDATA))
    print('length of EMGLABEL', len(EMGLABEL))

    # plt.plot(EMGDATA[0])
    # plt.title(EMGLABEL[0])
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(EMGDATA, EMGLABEL, test_size=0.3)

    print("length of X_train:", len(X_train))

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)

    # print(X_train.shape)
    # plt.subplot(4, 2, 1)
    # plt.plot(X_train[0, 0, 0, :])
    # plt.subplot(4, 2, 2)
    # plt.plot(X_train[0, 0, 1, :])
    # plt.subplot(4, 2, 3)
    # plt.plot(X_train[0, 0, 2, :])
    # plt.subplot(4, 2, 4)
    # plt.plot(X_train[0, 0, 3, :])
    # plt.subplot(4, 2, 5)
    # plt.plot(X_train[0, 0, 4, :])
    # plt.subplot(4, 2, 6)
    # plt.plot(X_train[0, 0, 5, :])
    # plt.subplot(4, 2, 7)
    # plt.plot(X_train[0, 0, 6, :])
    # plt.subplot(4, 2, 8)
    # plt.plot(X_train[0, 0, 7, :])
    # plt.show()

    X_train = torch.from_numpy(np.array(X_train)).type(torch.FloatTensor)
    X_test = torch.from_numpy(np.array(X_test)).type(torch.FloatTensor)
    y_train = torch.from_numpy(np.array(y_train))
    y_test = torch.from_numpy(np.array(y_test))

    # ----------------------------------

    X_train = X_train.reshape(-1, 8, 5000)
    X_test = X_test.reshape(-1, 8, 5000)



    train_Dataset = Data.TensorDataset(X_train, y_train)
    test_Dataset = Data.TensorDataset(X_test, y_test)


    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(
        dataset=train_Dataset,
        batch_size=96,
        shuffle=True)

    test_loader = Data.DataLoader(
        dataset=test_Dataset,
        batch_size=96,
        shuffle=True)



    cnn = CNN()
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    for epoch in range(30):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

            output = cnn(b_x)[0]                          # cnn output
            loss = loss_func(output, b_y)               # cross entropy loss
            optimizer.zero_grad()                       # clear gradients for this training step
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                correct = 0
                total = 0

                for step, (b_x, b_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader

                    test_output = cnn(b_x)[0]
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    right = torch.sum(pred_y == b_y).type(torch.FloatTensor)
                    total += b_y.size()[0]
                    correct += right
                accuracy = correct / total
                print("训练", epoch, "后，测试", total, "个数据, ", "准确率为：", accuracy.data)

    # 全部训练完成
    print("训练完成，开始测试：")

    correct = 0
    total = 0
    for step, (b_x, b_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader

        test_output = cnn(b_x)[0]
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        right = torch.sum(pred_y == b_y).type(torch.FloatTensor)
        total += b_y.size()[0]
        correct += right

    accuracy = correct / total

    print("训练", epoch + 1, "次后，测试", total, "个数据, ", "准确率为：", accuracy.data)

    torch.save(cnn, "PytorchModel_CNN_1D_norm.pkl")


