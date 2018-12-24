# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

# LAYER1_NODE = 500

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, is_train, regularizer):

    # 输入为28*28*1的原始图片像素；输出为28*28*32矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weights", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器，过滤器移动步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 最大池化层，过滤器边长为2. 全0填充，移动步长为2. 输出为14*14*32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')

    # 输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            "biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为64的过滤器，过滤器移动步长为1，全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[
                             1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 与layer2结构一样，输出为7*7*64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')

    # 转换为第五层 全连接层的输入格式（向量）
    pool_shape = pool2.get_shape().as_list()

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) #pool_shape[0]为一个batch的数据个数

    # 向量长度为7*7*64=3136, 输出是512向量
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        fc1_biases = tf.get_variable(
            "biases", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # 一般在全连接层，避免过拟合，仅训练时
        if is_train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 输出长度为10的向量
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable(
            "biases", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
