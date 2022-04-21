import os
import sys
import numpy as np
from dataset.mnist import load_mnist
import pickle

sys.path.append(os.pardir)


# 与门
def per_and(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1


# 与非门
def per_nand(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1


# 或门
def per_or(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1


# 异或门
def per_xor(x1, x2):
    return per_and(per_nand(x1, x2), per_or(x1, x2))


# 隐藏层激活函数sigmod,这里会出现RunTimeWraning报告,但是不打紧,这是numpy的报错
def per_sigmod(x):
    return 1 / (1 + np.exp(-x))


# 隐藏层激活函数ReLU
def per_relu(x):
    return np.maximum(0, x)


# 输出层激活函数,恒等函数,一般用在回归问题上
def identify_function(x):
    return x


# ----------------------------------- #
# 输出层激活函数,softmax,一般用在分类问题上
# 各个输出值受到所有输出值的影响
# 在视觉识别中多用softmax,因为其输出结果总和为1,可以理解成每个类的概率,概率越大,越符合预测
# ----------------------------------- #
def per_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 防止数据溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 神经系统处理的初始化,此处时导入已有的pkl文件的权重数据
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


# 神经网络处理中的向前处理操作
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = per_sigmod(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = per_sigmod(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)
    return y


# 数据获取
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


# 损失函数,均方误差函数
def mean_squared_error(y, t):
    return 0.5*np.sum((np.array(y)-np.array(t))**2)


# 损失函数,交叉熵误差,这个韩式只适用于单个数据输入
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(np.array(t)*np.log(np.array(y)+np.array(delta)))


# 交叉熵误差,mini_batch版本
def cross_entropy_error_mini(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.array(t)*np.log(np.array(y)+1e-7)) / batch_size


# 交叉熵误差,mini_batch_not_one_hot版本
def cross_entropy_error_mini_none(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
