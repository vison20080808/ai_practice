# https://www.jianshu.com/p/41466470b347
# 用tensorboard来看看我们的网络流吧！


import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):

    layer_name = 'layer%s' % n_layer

    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name+'/weights', Weights)
            print(layer_name, Weights.shape)  # (1, 10)
            print(layer_name, Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name+'/biases', biases)
            print(layer_name, biases.shape)  # (1, 10)
            print(layer_name, biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)  # (?, 10)
            print(layer_name, Wx_plus_b.shape)
            print(layer_name, Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
print(x_data.shape)  # (300, 1)

noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input') ##None表示给多少个sample都可以
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    test = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], shape=[1, 10])
    print('test.shape = ', test.shape)  # (1, 10)
    print('test = ', sess.run(test))  # [[1 2 3 4 5 6 7 8 9 0]]

    test2 = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype=np.float32)
    print('test2 = ', sess.run(test2))
    print('全部求平均值 = ', sess.run(tf.reduce_mean(test2)))
    print('按列求平均值 = ', sess.run(tf.reduce_mean(test2, 0)))  # 第一维
    print('按行求平均值 = ', sess.run(tf.reduce_mean(test2, 1)))  # 第二维

    # 1.2之前 tf.train.SummaryWriter("logs/",sess.graph)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)



