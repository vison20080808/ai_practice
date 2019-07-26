# https://www.jianshu.com/p/d25baccde6bc
# 更进一步，使用LSTM实现对手写数字识别
# 对于一张28*28维的图片来说，我们可以把每一行当成一次输入，序列长度为行数，同时，我们取最后一个输出的输出作为预测结果。


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('../data', one_hot=True)

lr = 0.001

training_iters = 1000000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

xs = tf.placeholder(tf.float32, [None, n_inputs, n_steps])
ys = tf.placeholder(tf.float32, [None, 10])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    print(X)  # shape=(?, 28)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    print(X_in)  # shape=(?, 128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    print(X_in)  # shape=(?, 28, 128)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # final_states[1]就是短时记忆h
    results = tf.matmul(final_states[1], weights['out']) + biases['out']

    return results

prediction = RNN(xs, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(ys, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0

    # validate_feed = {xs: np.reshape(mnist.validation.images, (mnist.validation.num_examples, n_inputs, n_steps)),
    #                  ys: mnist.validation.labels}
    #
    # test_feed = {xs: np.reshape(mnist.test.images, (mnist.test.num_examples, n_inputs, n_steps)),
    #              ys: mnist.test.labels}

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # print('batch_xs=', batch_xs, '; batch_ys=', batch_ys)  # [[], []]

        #一个step是一行
        batch_xs = batch_xs.reshape([batch_size, n_inputs, n_steps])  #[[[ ]]]
        # print('after reshape batch_xs=', batch_xs)

        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys}))

        # if step % 1000 == 0:
        #     validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        #     print("After %d training step(s), validation accuracy "
        #           "using average model is %g " % (step, validate_acc))

        step += 1

    # test_acc = sess.run(accuracy, feed_dict=test_feed)
    # print("After %d training step(s), test accuracy using average "
    #       "model is %g" % (training_iters, test_acc))




