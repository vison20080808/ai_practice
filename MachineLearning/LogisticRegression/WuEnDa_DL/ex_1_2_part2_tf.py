
import numpy as np
import h5py

import matplotlib.pyplot as plt

import tensorflow as tf


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b∗c∗d, a)
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
# print ("train_set_x[0]: " + str(train_set_x[0, :]))
# print ("train_set_y[0]: " + str(train_set_y[0, :]))

w = tf.Variable(tf.zeros([train_set_x.shape[0], 1], dtype=tf.float64), dtype=tf.float64)
b = tf.Variable(tf.zeros([1, 1], dtype=tf.float64), dtype=tf.float64)


# train_set_x = train_set_x.T
# test_set_x = test_set_x.T
# print ("train_set_x shape: " + str(train_set_x.shape))
# print ("test_set_x shape: " + str(test_set_x.shape))
#
#
# w = tf.Variable(tf.zeros([m_train, 1], dtype=tf.float64), dtype=tf.float64)
# b = tf.Variable(tf.zeros([1, 1], dtype=tf.float64), dtype=tf.float64)

A = 1 / (1 + tf.exp(-tf.matmul(tf.transpose(w), train_set_x) + b))

cost = tf.reduce_mean(- train_set_y * tf.log(A) - (1 - train_set_y) * tf.log(1 - A))

# cost = -1.0 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

# dw = 1.0 / m * np.dot(X, (A - Y).T)
# db = 1.0 / m * np.sum(A - Y)

train = tf.train.GradientDescentOptimizer(0.005).minimize(cost)
# train = tf.train.AdamOptimizer(0.005).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2000):
    sess.run(train)

    w_res = sess.run(w).flatten()
    b_res = sess.run(b).flatten()
    print(step, w_res, b_res)


def sigmoid(x):
    # return 1.0 / (1 + 1/math.exp(x))
    # return 1.0 / (1 + 1 / np.exp(x))
    return 1.0 / (1 + np.exp(-x))

def predict(w, b, X):
    m = X.shape[1]

    Y_prediction = np.zeros((1, m))

    # print(w.shape)
    w = w.reshape(X.shape[0], 1)
    # print(w.shape)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

Y_prediction_test = predict(w_res, b_res, test_set_x)
Y_prediction_train = predict(w_res, b_res, train_set_x)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
# GradientDescentOptimizer:
# train accuracy: 99.04306220095694 %
# test accuracy: 70.0 %

# AdamOptimizer
# train accuracy: 100.0 %
# test accuracy: 72.0 %

