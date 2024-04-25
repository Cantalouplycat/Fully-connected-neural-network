"""
File: NN_basis.py
Author: DM
Date: 2024/1/30,
Description: 单隐藏层全连接神经网络类.
"""
import math
import numpy as np


# 激活函数
def Sigmoid(x):
    x1 = math.pow(math.e, -0.5 * x)
    y = 1 / (1 + x1)
    return y


# 神经网络类
class Mnn_net:
    # 构造函数，在创建对象时被调用
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化损失函数,输入层,隐藏层,输出层节点数
        self.Ek = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 初始化权重
        self.v = np.random.uniform(low=-0.5, high=0.5,
                                   size=(hidden_size, input_size))
        self.b = np.random.rand(hidden_size, 1)
        self.r = np.random.rand(hidden_size, 1)
        self.w = np.random.uniform(low=-0.5, high=0.5,
                                   size=(output_size, hidden_size))
        self.c = np.random.rand(output_size, 1)
        self.y = np.random.rand(output_size, 1)

    # 神经网络输出
    def C_output(self, x):
        x = x.reshape(self.input_size, 1)
        self.b = self.v @ x - self.r
        self.b = np.vectorize(Sigmoid)(self.b)
        self.y = self.w @ self.b - self.c
        self.y = np.vectorize(Sigmoid)(self.y)

    # 神经网络更新权重
    def Update(self, x, yk, n1, n2):
        x = x.reshape(self.input_size, 1)
        yk = yk.reshape(self.output_size, 1)
        g = self.y * (1 - self.y) * (self.y - yk)
        dw = - n1 * g @ self.b.T
        dc = n1 * g
        dv = -n2 * (self.b * (1 - self.b) * (self.w.T @ g)) @ x.T
        dr = n2 * (self.b * (1 - self.b) * (self.w.T @ g))
        self.w = self.w + dw
        self.c = self.c + dc
        self.v = self.v + dv
        self.r = self.r + dr
        self.Ek = 0.5 * np.sum(np.square(self.y - yk))

    # 导入神经网络权重
    def Model_evaluation(self, v, r, w, c):
        self.v = v
        self.r = r
        self.w = w
        self.c = c
