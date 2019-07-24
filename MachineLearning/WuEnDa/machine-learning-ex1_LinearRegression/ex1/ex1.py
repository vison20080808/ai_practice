

from numpy import mat, shape, zeros, arange

import matplotlib.pyplot as plt

def loadDataSet(file):
    dataMat = []
    labelMat = []
    with open(file) as f:
        for line in f.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([1.0, float(lineArr[0])])  # 增加了第一列均为1.0
            labelMat.append(float(lineArr[1]))

    return dataMat, labelMat


def gradDescent(X, y):
    X_matrix = mat(X)
    y_matrix = mat(y).reshape(-1, 1)  # 转置

    print(shape(X_matrix))  # (97, 2)
    print(shape(y_matrix))  # (97, 1)

    m, n = shape(X_matrix)  # m行数据，n维特征
    alpha = 0.01  # 学习率
    iterations = 1500  # 最大迭代次数
    theta = zeros((n, 1))

    for k in range(iterations):
        error = X_matrix * theta - y_matrix  # h(x) - y
        theta = theta - alpha * X_matrix.transpose() * error / m  # w = w - a(h - y)x/m

    return theta

if __name__ == '__main__':
    ex1_file = './ex1data1.txt'
    X, y = loadDataSet(ex1_file)

    theta = gradDescent(X, y)
    print(theta)
    # [[-3.63029144]
    #  [ 1.16636235]]

    print(shape(theta))  # (2, 1)

    slope = theta[1, 0]
    print(slope)

    intercept = theta[0, 0]
    print(intercept)

    xx = arange(1, 40, 1)
    yy = intercept + slope * xx

    x = mat(X)[:, 1]
    y = mat(y).reshape(-1, 1)
    print(shape(x))  # (97, 1)
    print(shape(y))  # (97, 1)
    plt.plot(x, y, '.')
    plt.plot(xx, yy, '--')
    plt.xlim(0, 40)
    plt.ylim(-10, 40)
    plt.show()









