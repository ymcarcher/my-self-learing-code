import os
import sys
import numpy as np
from perceptron_file.perceptron import cross_entropy_error, per_softmax
sys.path.append(os.pardir)


class SimpleNet:
    def __init__(self):
        # 使用高斯分布初始化
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = per_softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
