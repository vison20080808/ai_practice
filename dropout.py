# 使用dropout来避免过拟合吧！
# https://www.jianshu.com/p/4f1b525ddf86


import tensorflow as tf
import numpy as np


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


digits = load_digits()
X = digits.data
Y = digits.target
print(Y.shape)
print(Y)
Y = LabelBinarizer().fit_transform(Y)
print(Y.shape)
print(Y)

trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.3)


def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):

    layer_name = 'layer%s' % n_layer

    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name+'/weights', Weights)
            print(layer_name, Weights.shape)
            print(layer_name, Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name+'/biases', biases)
            print(layer_name, biases.shape)
            print(layer_name, biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            print(layer_name, Wx_plus_b.shape)
            print(layer_name, Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

            # 这里的output是一个二维的，所以每一步对应一个线（或者说小的矩形，颜色越深的地方表示这个地方的数越多，可以认为纵向上表示train到这一步的时候的一个数据分布
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 64])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs, 64, 50, 1, activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 2, activation_function=tf.nn.softmax)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    # 因为cross_entropy是一个标量，所以定义tf.summary.scalar
    tf.summary.scalar('loss', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 合并所有的summary
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logs/a003/train/', sess.graph)
    test_writer = tf.summary.FileWriter('logs/a003/test/', sess.graph)

    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: trainx, ys: trainy, keep_prob: 0.5})

        if i % 50 == 0:
            print(sess.run(cross_entropy, feed_dict={xs: trainx, ys: trainy}))
            # 这里要运行merged
            train_loss = sess.run(merged, feed_dict={xs: trainx, ys: trainy, keep_prob: 0.5})
            test_loss = sess.run(merged, feed_dict={xs: testx, ys: testy, keep_prob: 0.5})

            train_writer.add_summary(train_loss, i)
            test_writer.add_summary(test_loss, i)









