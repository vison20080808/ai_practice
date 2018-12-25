# -*- coding: utf-8 -*-

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import mnist_cnn_inference as mnist_inference

import mnist_cnn_train as mnist_train
import numpy as np


EVEL_INTERVAL_SECS = 20



def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # x = tf.placeholder(
        #    tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')

        x = tf.placeholder(tf.float32, [mnist.test.num_examples,
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS],
                           name='x-input')

        reshaped_xs = np.reshape(mnist.test.images,
                                 (mnist.test.num_examples, mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))

        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: reshaped_xs,
                         y_: mnist.test.labels}

        y = mnist_inference.inference(x, False, None)
        softmax_out = tf.nn.softmax(y, name='out_softmax')  # 输出节点
        correct_prediction = tf.equal(
            tf.argmax(softmax_out, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1]
                    accuracy_score = sess.run(
                        accuracy, feed_dict=validate_feed)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                          " After %s training step(s), test accuracy "
                          "= %g " % (global_step, accuracy_score))
                    # 2018-12-19 19:32:08 After 21501 training step(s), test accuracy = 0.9943
                else:
                    print('No checkpoint file found')
                    return

                time.sleep(EVEL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets('../data', one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
