
import tensorflow as tf
import numpy as np

X = [1, 2]
state = [0.0, 0.0]

w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell

    state = np.tanh(before_activation)

    final_output = np.dot(state, w_output) + b_output

    print('step', i)
    print('before activation:', before_activation)
    print('state', state)
    print('output: ', final_output)



# +++++++LSTM+++++++
# lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
# state = lstm.zero_state(batch_size, tf.float32)
# loss = 0.0
#
# for i in range(num_steps):
#     # 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
#     if i > 0 : tf.get_variable_scope().reuse_variables()
#
#     lstm_output, state = lstm(current_input, state)
#     # outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
#     final_output = fully_connected(lstm_output)
#
#     loss += calc_loss(final_output, expected_output)


# ++++++++ Deep RNN ++++++++++++
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
# stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size) for _ in range(number_of_layers)])
#
# state = stacked_lstm.zero_state(batch_size=, tf.float32)
#
# for i in range(num_steps):
#     if i > 0: tf.get_variable_scope().reuse_variables()
#     stacked_lstm_output, state = stacked_lstm(current_input, state)
#     final_output = fully_connected(stacked_lstm_output)
#     loss += calc_loss(final_output, expected_output)


# ++++++++ RNN Dropout ++++++++++
# stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size)) for _ in range(number_of_layers)])



########## 利用RNN实现对sinx的取值的预测 ############
import tensorflow as tf
import numpy as np

import matplotlib as mpl
# mpl.use('Agg')

from matplotlib import pyplot as plt

HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数

TIMESTEPS = 10  # 训练序列长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01  #采样间隔


def generate_data(seq):
    X = []
    Y = []

    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def lstm_model(X, Y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
                                       for _ in range(NUM_LAYERS)])

    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    output = outputs[:, -1, :]  # -1表示最后一个时刻的输出结果

    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn = None)

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=Y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer='Adagrad', learning_rate = 0.1)

    return predictions, loss, train_op


def train(sess, train_X, train_Y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, Y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model'):
        predictions, loss, train_op = lstm_model(X, Y, True)

    sess.run(tf.global_variables_initializer())

    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print('train step: ' + str(i) + ", loss:" + str(l))


def run_eval(sess, test_X, test_Y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_Y))
    ds = ds.batch(1)
    X, Y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model', reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    predictions = []
    labels = []

    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, Y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis = 0))
    print('Mean Square Error is: %f' % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP

train_X, train_Y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_Y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

with tf.Session() as sess:
    train(sess, train_X, train_Y)
    run_eval(sess, test_X, test_Y)













































