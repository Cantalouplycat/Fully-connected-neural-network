"""
File: model_train.py
Author: DM
Date: 2024/1/30,
Description: 神经网络训练程序.
"""
import NN_basis as Mnn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 定义训练数据类
class Date:
    def __init__(self, date, idenx):
        self.date = date
        self.idenx = idenx


# 导入数据标签
with open('./Image_enhance/idenx.txt', 'r') as file:
    content = file.read()
length = int(len(content)/4)

# 导入训练数据及测试数据
date_train = []
date_test = []
for i in range(length):
    date_train.append(Date(plt.imread("./Image_enhance/" + str(i) + ".png"), content[2 * i]))
for i in range(length):
    date_test.append(Date(plt.imread("./Image_enhance/" + str(i + length) + ".png"), content[2 * (i + length)]))

# 定义神经网络
Mynet = Mnn.Mnn_net(input_size=784, hidden_size=300, output_size=10)

# 设置训练轮数
epochs_n = 100

# 初始化混淆矩阵
conf_matrix_test = np.zeros((10, 10), dtype=int)

# 训练
for epoch in range(epochs_n):
    # 初始化训练正确数,测试正确数,损失函数
    running_correct = 0
    testing_correct = 0
    running_loss = 0
    # 训练
    for i in range(4000):
        # 将训练图片线性化
        x = np.ravel(date_train[i].date)
        # 将图片标签转化为神经网络输出格式
        y = np.zeros([1, 10])
        n = int(date_train[i].idenx)
        y[0][n] = 1
        # 计算神经网络输出
        Mynet.C_output(x)
        # 更新权重,学习率为0.1,0.1
        Mynet.Update(x, y, 0.1, 0.1)
        # 计算损失函数
        running_loss += Mynet.Ek
        # 计算训练精度
        if np.argmax(Mynet.y) == n:
            running_correct = running_correct + 1
    # 测试
    for i in range(4000):
        # 将测试图片线性化
        x = np.ravel(date_test[i].date)
        # 将图片标签转化为神经网络输出格式
        y = np.zeros([1, 10])
        n = int(date_test[i].idenx)
        y[0][n] = 1
        # 计算神经网络输出
        Mynet.C_output(x)
        # 以最后一轮测试结果绘制混淆矩阵
        if epoch == epochs_n - 1:
            conf_matrix_test[n, np.argmax(Mynet.y)] += 1
        # 计算测试精度
        if np.argmax(Mynet.y) == n:
            testing_correct = testing_correct + 1
        # 打印训练信息
        print("\rEpoch{}/{}  损失函数值:{:.4f},训练精度:{:.4f}%, 测试精度:{:.4f}%"
              .format(epoch + 1, epochs_n, running_loss / 4000,
                      100 * running_correct / 4000,
                      100 * testing_correct / 4000), end='')
    print(" ")
# 保存神经网络权重
np.savez('My_model_new.npz', v=Mynet.v, r=Mynet.r, w=Mynet.w, c=Mynet.c)
# 绘制混淆矩阵
# plt.subplot(1, 1, 1)
# sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
# plt.title('Confusion Matrix')
# plt.show()
