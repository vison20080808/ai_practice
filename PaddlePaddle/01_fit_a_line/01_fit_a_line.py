
# 经典的线性回归（Linear Regression）
# 波士顿房价数据集

# 对于线性回归模型来讲，最常见的损失函数就是均方误差（Mean Squared Error， MSE）

from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import math
import sys


# http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/quick_start/fit_a_line/README.cn.html

# 配置数据提供器(Datafeeder)

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500
    ), batch_size=BATCH_SIZE
)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.test(), buf_size=500
    ), batch_size=BATCH_SIZE
)


# 配置训练程序
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(cost)


# Optimizer Function 配置
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

test_program = main_program.clone(for_test=True)



# 定义运算场所
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

exe = fluid.Executor(place)

# 还可以通过画图，来展现训练进程
from paddle.utils.plot import Ploter

train_prompt = 'Train cost'
test_prompt = 'Test cost'
# plot_prompt = Ploter(train_prompt, test_program)



# 创建训练过程
num_epochs = 100

def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0

    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1

    return [x_d / count for x_d in accumulated]


# 训练主循环
params_dirname = '01_fit_a_line.inference.model'
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
naive_exe = fluid.Executor(place)
naive_exe.run(startup_program)

step = 0
exe_test = fluid.Executor(place)

for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value,  = exe.run(main_program, feed=feeder.feed(data_train),
                                   fetch_list=[avg_loss])

        if step % 10 == 0:
            # plot_prompt.append(train_prompt, step, avg_loss_value[0])
            # plot_prompt.plot()
            print('%s, Step %d, Cost %f' % (train_prompt, step, avg_loss_value[0]))
            # Train cost, Step 2090, Cost 36.129532

        if step % 100 == 0:
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            # plot_prompt.append(test_prompt, step, test_metics[0])
            # plot_prompt.plot()
            print('\n%s, Step %d, Cost %f\n\n' % (test_prompt, step, test_metics[0]))
            # Test cost, Step 2000, Cost 22.764661

            if test_metics[0] < 10.0:
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit('Got NaN loss, training failed.')

    if params_dirname is not None:
        fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)



# 预测

infer_exe = fluid.Executor(place)
infer_scope = fluid.core.Scope()

with fluid.scope_guard(infer_scope):
    [infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)

    batch_size = 10

    infer_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=batch_size)

    infer_data = next(infer_reader())

    infer_feat = numpy.array([data[0] for data in infer_data]).astype('float32')
    infer_label = numpy.array([data[1] for data in infer_data]).astype('float32')

    assert feed_target_names[0] == 'x'

    results = infer_exe.run(infer_program, feed={feed_target_names[0]: numpy.array(infer_feat)},
                            fetch_list=fetch_targets)

    print('\ninfer results: (House Price)')
    for idx, val in enumerate(results[0]):
        print('%d: %.2f' % (idx, val))
    # infer results: (House Price)
    # 0: 14.59
    # 1: 15.18
    # 2: 14.41
    # 3: 16.59
    # 4: 14.85
    # 5: 15.87
    # 6: 15.87
    # 7: 15.05
    # 8: 12.16
    # 9: 15.01


    print('\nground truth:')

    for idx, val in enumerate(infer_label):
        print('%d: %.2f' % (idx, val))
    # ground truth:
    # 0: 8.50
    # 1: 5.00
    # 2: 11.90
    # 3: 27.90
    # 4: 17.20
    # 5: 27.50
    # 6: 15.00
    # 7: 17.20
    # 8: 17.90
    # 9: 16.30


