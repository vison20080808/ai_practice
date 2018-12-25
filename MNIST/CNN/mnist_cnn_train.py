# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_cnn_inference as mnist_inference

from tensorflow.python.framework import graph_util
import numpy as np
import time

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRANING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')

        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    softmax_out = tf.nn.softmax(y, name='out_softmax')  # 输出节点

    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=softmax_out,  labels=tf.argmax(y_, 1))

        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate)\
            .minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(".log", tf.get_default_graph())

    #correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRANING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))


            if i % 500 == 0:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, loss_value, step = sess.run(
                    [train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys},
                options=run_options, run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%03d' % i)

                #validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                # print("After %d training step(s), loss on training "
                #       "batch is %g." % (step, loss_value))

                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                      " After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                              MODEL_NAME), global_step=global_step)
            else:
                _, loss_value, step = sess.run(
                    [train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ["out_softmax"])
        with tf.gfile.FastGFile('mnist.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets('../data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
