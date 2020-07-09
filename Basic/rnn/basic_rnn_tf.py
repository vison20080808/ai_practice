
import numpy as np
import tensorflow as tf

n_x = 3  # n_inputs
n_a = 5  # n_units
batch_size = 4

x0 = tf.placeholder(tf.float32, [None, n_x])
x1 = tf.placeholder(tf.float32, [None, n_x])

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_a)
a0 = cell.zero_state(batch_size, dtype=tf.float32)  # init_state
outputs, a = tf.nn.static_rnn(cell, [x0, x1], initial_state=a0)

y0, y1 = outputs

init = tf.global_variables_initializer()

x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    sess.run(init)

    y0_val, y1_val = sess.run([x0, x1], feed_dict={x0: x0_batch, x1: x1_batch})

print('y0_val = ', y0_val)
print('y1_val = ', y1_val)


Wx = tf.Variable(tf.random_normal([n_x, n_a], dtype=tf.float32), name="Wx")
Wh = tf.Variable(tf.random_normal([n_a, n_a], dtype=tf.float32), name="Wh")
b = tf.Variable(tf.zeros([n_a, ], dtype=tf.float32), name="b")

h0 = tf.tanh(tf.matmul(x0, Wx) + b)
h1 = tf.tanh(tf.matmul(x1, Wx) + tf.matmul(h0, Wh) + b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    h0_val, h1_val = sess.run([h0, h1], feed_dict={x0: x0_batch, x1: x1_batch})

print("the output of h0_val are")
print(h0_val)

print("the output of h1_val are")
print(h1_val)