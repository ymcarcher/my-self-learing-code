from perceptron_file.perceptron import *
import numpy as np

from simplenet.numerical_unit import numerical_grad

sys.path.append(os.pardir)


# 两层的神经网络类
class TwoLayerOut:
    """
    input_size          输入层神经元个数
    hidden_size         中间层神经元个数
    output_size         输出层神经元个数
    weight_init_std
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 创建字典
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 进行预测
    def predict(self, x):
        print(1)
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = per_sigmod(a1)
        a2 = np.dot(z1, W2) + b2
        y = per_sigmod(a2)

        return y

    """
    x:输入数据,t:监督数据
    损失函数
    """

    def loss(self, x, t):
        print(2)
        y = self.predict(x)

        return cross_entropy_error(y, t)

    """
    精准度计算
    """

    def accuracy(self, x, t):
        print(3)
        y = self.predict(x)
        # argmax求最大值的索引
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算权重参数的梯度
    def gradient(self, x, t):
        print(4)
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_grad(loss_W, self.params['W1'])
        grads['b1'] = numerical_grad(loss_W, self.params['b1'])
        grads['W2'] = numerical_grad(loss_W, self.params['W2'])
        grads['b2'] = numerical_grad(loss_W, self.params['b2'])

        return grads
