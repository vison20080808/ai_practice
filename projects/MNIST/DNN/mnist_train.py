# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRANING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.interence(x, regularizer)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,  labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    #correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # validate_feed = {x: mnist.validation.images,
        #                y_: mnist.validation.labels}

        #test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRANING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                #validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                              MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('../data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
