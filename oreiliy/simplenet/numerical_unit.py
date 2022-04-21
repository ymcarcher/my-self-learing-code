import numpy as np


# 求数值微分,f是函数输入
def numerical_diff(f, x):
    h = 1e4
    return (f(x + h) - f(x - h)) / (2 * h)


# 梯度,f是函数输入
def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成和x形状相同的数组

    # 创建迭代器,存放的是x,方便后面调用x的值
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        # 计算f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        # 计算f(x-h)
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原值
        x[idx] = tmp_val
        # 调用下一个值
        it.iternext()

    return grad


# 下降梯度法求梯度
def grad_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_grad(f, x)
        x -= lr * grad
    return x
