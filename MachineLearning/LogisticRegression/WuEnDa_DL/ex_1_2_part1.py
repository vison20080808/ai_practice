
import math

import numpy as np

def sigmoid(x):
    # return 1.0 / (1 + 1/math.exp(x))
    # return 1.0 / (1 + 1 / np.exp(x))
    return 1.0 / (1 + np.exp(-x))

print(sigmoid(3))
print(sigmoid(0))

x = np.array([1, 2, 3])
print(sigmoid(x))


def sigmoid_derivative(x):
    s = 1.0 / (1 + 1 / np.exp(x))
    return s * (1 - s)

print(sigmoid_derivative(x))


def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))


def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)  # 列向量

    print(x.shape)

    print(x_norm)
    print(x_norm.shape)


    x = x / x_norm  #利用numpy的广播，用矩阵与列向量相除。

    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


def softmax(x):
    x_exp = np.exp(x)

    x_row_sum = np.sum(x_exp, axis=1, keepdims=True)

    print('x.shape = ', x.shape)
    print('x_row_sum = ', x_row_sum)
    print('x_row_sum.shape = ', x_row_sum.shape)

    s = x_exp / x_row_sum

    print('s.shape = ', s.shape)
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))



def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))


def L2(yhat, y):
    # h = y - yhat
    # loss = np.dot(h, h.T)

    loss = np.sum(np.power(y - yhat, 2))

    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))

