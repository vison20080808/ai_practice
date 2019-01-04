



print('===========part1===========')
# NumPy 与 神经网络
# https://www.numpy.org.cn/article/advanced/neural_network_with_numpy.html

# 代码来源：https://github.com/FlorianMuellerklein/Machine-Learning

import math
import random
import numpy as np

np.seterr(all='ignore')


# sigmoid transfer function
# IMPORTANT: when using the logit (sigmoid) transfer function for the output layer make sure y values are scaled from 0 to 1
# if you use the tanh for the output then you should scale between -1 and 1
# we will use sigmoid for the output layer and tanh for the hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)


# using tanh over logistic sigmoid is recommended
def tanh(x):
    return math.tanh(x)


# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y * y


class MLP_NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network, adapted and from the book 'Programming Collective Intelligence' (http://shop.oreilly.com/product/9780596529321.do)
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm has been generalized to be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """

    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

        # initialize arrays
        self.input = input + 1  # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1 / 2)
        output_range = 1.0 / self.hidden ** (1 / 2)
        self.wi = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
        self.wo = np.random.normal(loc=0, scale=output_range, size=(self.hidden, self.output))

        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input - 1):  # -1 is to avoid the bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def train(self, patterns):
        # N: learning rate
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            if i % 10 == 0:
                print('error %-.5f' % error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (
                        self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions


def demo():
    """
    run NN demo on the digit recognition dataset from sklearn
    """

    def load_data():
        data = np.loadtxt('Data/sklearn_digits.csv', delimiter=',')

        # first ten values are the one hot encoded y (target) values
        y = data[:, 0:10]
        # y[y == 0] = -1 # if you are using a tanh transfer function make the 0 into -1
        # y[y == 1] = .90 # try values that won't saturate tanh

        data = data[:, 10:]  # x data
        # data = data - data.mean(axis = 1)
        data -= data.min()  # scale the data so values are between 0 and 1
        data /= data.max()  # scale

        out = []
        print(data.shape)

        # populate the tuple list with the data
        for i in range(data.shape[0]):
            fart = list((data[i, :].tolist(), y[i].tolist()))  # don't mind this variable name
            out.append(fart)

        return out

    X = load_data()

    print(X[9])  # make sure the data looks right

    NN = MLP_NeuralNetwork(64, 100, 10, iterations=50, learning_rate=0.5, momentum=0.5, rate_decay=0.01)

    NN.train(X)

    NN.test(X)


# if __name__ == '__main__':
# demo()




print('===========part2===========')
# https://www.numpy.org.cn/article/advanced/numpy_array_programming.html

# 使用NumPy进行数组编程
# #目录
# 进入状态：介绍NumPy数组
# 什么是矢量化？
# 计数:-简单的如：1,-2,-3…
# 买低，卖高
# Intermezzo：理解轴符号
# 广播
# 矩阵编程实际应用：示例
# 聚类算法
# 摊还（分期）表
# 图像特征提取
# 临别赠言：不要过度优化
# 更多资源


# 在计算方面，实际上有三个概念为NumPy提供了强大的功能：
# 矢量化
# 广播
# 索引


import numpy as np

arr = np.arange(36).reshape(3, 4, 3)
print(arr)

# NumPy中的向量化操作将内部循环委托给高度优化的C和Fortran函数，从而实现更清晰，更快速的Python代码。
# 计数: 简单的如：1, 2, 3…
# 作为示例，考虑一个True和False的一维向量，你要为其计算序列中“False to True”转换的数量：
np.random.seed(444)
x = np.random.choice([False, True], size=100000)
print(x)

# 传统Python方案：
def count_transitions(x) -> int:
    count = 0
    for i, j in zip(x[:-1], x[1:]):
        if j and not i:
            count += 1
    return count

print(count_transitions(x))

# 矢量化方案：
print(np.count_nonzero(x[:-1] < x[1:]))

# 性能对比：
from timeit import timeit
# setup = 'from __main__ import count_transitions, x; import numpy as np'
# num = 1000
# t1 = timeit('count_transitions(x)', setup=setup, number=num)
# t2 = timeit('np.count_nonzero(x[:-1] < x[1:])', setup=setup, number=num)
# print('Speed t1 = {:f}; t2 = {:f}'.format(t1, t2))  # Speed t1 = 9.178686; t2 = 0.114566
# print('Speed diff: {:0.1f}x'.format(t1 / t2))  # Speed diff: 80.1x



# 假定一只股票的历史价格是一个序列，假设你只允许进行一次购买和一次出售，那么可以获得的最大利润是多少？例如，假设价格=(20，18，14，17，20，21，15)，最大利润将是7，从14买到21卖。

# 迭代序列一次，找出每个价格和运行最小值之间的差异。

def profit(prices):
    max_px = 0
    min_px = prices[0]
    for px in prices[1:]:
        min_px = min(min_px, px)
        max_px = max(px - min_px, max_px)
    return max_px


prices = (20, 18, 14, 17, 20, 21, 15)
print(profit(prices))

# NumPy方案
def profit_with_numpy(prices):
    prices = np.asarray(prices)
    return np.max(prices - np.minimum.accumulate(prices))

print(profit_with_numpy(prices))

# 对比
# seq = np.random.randint(0, 100, size=100000)
# setup = ('from __main__ import profit_with_numpy, profit, seq; import numpy as np')
# num = 250
# pytime = timeit('profit(seq)', setup=setup, number=num)
# nptime = timeit('profit_with_numpy(seq)', setup=setup, number=num)
# print('Speed pytime = {:f}; nptime = {:f}'.format(pytime, nptime))  # Speed pytime = 17.227936; nptime = 0.129909
# print('Speed difference: {:0.1f}x'.format(pytime / nptime))  # Speed difference: 132.6x
# NumPy不仅可以委托给C，而且通过一些元素操作和线性代数，它还可以利用多线程中的计算。


# Intermezzo：理解轴符号
# AXIS关键字指定将折叠的数组的维度，而不是将要返回的维度。因此，指定Axis=0意味着第一个轴将折叠：对于二维数组，这意味着每列中的值将被聚合。
# 换句话说，如果将AXIS=0的数组相加，则会使用按 列 计算的方式 折叠 数组的 行 。


# 广播机制：
# 想要减去数组的每个列的平均值，元素的平均值：
sample = np.random.normal(loc=[2., 20.], scale=[1., 3.5], size=(3, 2))
print(sample)

mu = np.mean(sample, axis=0)
print(mu)

print('sample:', sample.shape, '| means: ', mu.shape)
print(sample - mu, (sample - mu).shape)  # 较小的数组被“拉伸”，以便从较大的数组的每一行中减去它：
# 技术细节：较小的数组或标量不是按字面意义上在内存中展开的：重复的是计算本身。

# 这扩展到标准化每个列，使每个单元格相对于其各自的列具有z-score：
print(sample.std(axis=0))
print((sample - sample.mean(axis=0)) / sample.std(axis=0))

# 要减去行最小值，该怎么办？
# sample - sample.min(axis=1)  # ValueError: operands could not be broadcast together with shapes (3,2) (3,)
# 问题是，较小的数组，在其目前的形式，不能“伸展”，以形状与样本兼容。实际上，你需要扩展它的维度，以满足上面的广播规则：
print(sample.min(axis=1)[:, None])  # 注意: [:, None]是一种扩展数组维度的方法，用于创建长度为1的轴。np.newaxis是None的别名。
print(sample - sample.min(axis=1)[:, None])

# 如果以下规则产生有效结果，则一组数组被称为“可广播”到相同的形状，这意味着 以下之一为真 时：
#
# 1. 矩阵都具有完全相同的形状。
# 2. 矩阵都具有相同数量的维度，每个维度的长度是公共长度或1。
# 3. 具有太少尺寸的矩列可以使其形状前面具有长度为1的尺寸以满足属性＃2。


# 矩阵编程实际应用：示例
# 聚类算法：簇的质心、欧几里得距离
# 摊还（分期）表：
# 图像特征提取： # https://scikit-learn.org/stable/modules/feature_extraction.html#image-feature-extraction

