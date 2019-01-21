

# 基于MNIST数据训练一个分类器

# Softmax回归(Softmax Regression)

# N 维结果向量经过softmax（激活函数）将归一化为 N 个[0,1]范围内的实数值，分别表示该样本属于这 N 个类别的概率。

# 在分类问题中，我们一般采用交叉熵代价损失函数（cross entropy loss）

# 多层感知器(Multilayer Perceptron, MLP)

# 在多层感知器模型中，将图像展开成一维向量输入到网络中，忽略了图像的位置和结构信息，
# 而卷积神经网络(Convolutional Neural Network, CNN) 能够更好的利用图像的结构信息。LeNet-5是一个较简单的卷积神经网络。

# 卷积层的参数较少，是由卷积层的主要特性即 局部连接 和 共享权重 所决定。卷积是线性操作，并具有平移不变性（shift-invariant）。
# 池化是非线性下采样的一种形式，主要作用是通过减少网络的参数来减小计算量，并且能够在一定程度上控制过拟合。通常在卷积层的后面会加上一个池化层。池化包括最大池化、平均池化等。

# 常见激活函数：sigmoid激活函数、tanh激活函数、ReLU激活函数
# 实际上，tanh函数只是规模变化的sigmoid函数，将sigmoid函数值放大2倍之后再向下平移1个单位：tanh(x) = 2sigmoid(2x) - 1


# Fluid API是最新的 PaddlePaddle API。它在不牺牲性能的情况下简化了模型配置。
# 下面是快速的 Fluid API 概述。
#
# inference_program：指定如何从数据输入中获得预测的函数。 这是指定网络流的地方。
#
# train_program：指定如何从 inference_program 和标签值中获取 loss 的函数。 这是指定损失计算的地方。
#
# optimizer_func: “指定优化器配置的函数。优化器负责减少损失并驱动培训。Paddle 支持多种不同的优化器。
#
# Trainer：PaddlePaddle Trainer 管理由 train_program 和 optimizer 指定的训练过程。 通过 event_handler 回调函数，用户可以监控培训的进展。
#
# Inferencer：Fluid inferencer 加载 inference_program 和由 Trainer 训练的参数。 然后，它可以推断数据和返回预测。


from __future__ import print_function

import os
from PIL import Image
import numpy
import paddle
import paddle.fluid as fluid


# Program Functions 配置
# 想用这个程序来演示三个不同的分类器，每个分类器都定义为 Python 函数。

def softmax_regression():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict

# 实现了一个含有两个隐藏层（即全连接层）的多层感知器。其中两个隐藏层的激活函数均采用ReLU，输出层的激活函数用Softmax。
def multilayer_perception():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    hidden = fluid.layers.fc(input=img, size=200,  act='relu')
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction

# 卷积神经网络LeNet-5: 输入的二维图像，首先经过两次卷积层到池化层，再经过全连接层，最后使用以softmax为激活函数的全连接层作为输出层
def convolutional_n_n():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')

    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img, filter_size=5, num_filters=20, pool_size=2, pool_stride=2, act='relu'
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=conv_pool_1, filter_size=5, num_filters=50,
                                                  pool_size=2, pool_stride=2, act='relu')

    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')

    return prediction


# Train Program 配置
# 注意: 训练程序应该返回一个数组，第一个返回参数必须是 avg_cost。训练器使用它来计算梯度。
def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    prediction = softmax_regression()
    prediction = multilayer_perception()
    prediction = convolutional_n_n()

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return prediction, [avg_cost, acc]


# Optimizer Function 配置
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


# 数据集 Feeders 配置

# paddle.dataset.mnist.train()和paddle.dataset.mnist.test()分别做训练和测试数据集。
# 这两个函数各自返回一个reader——PaddlePaddle中的reader是一个Python函数，每次调用的时候返回一个Python yield generator。
# shuffle是一个reader decorator
# batch是一个特殊的decorator， 一个batched reader每次yield一个minibatch。

BATCH_SIZE = 64

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500
    ), batch_size=BATCH_SIZE
)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.test(), buf_size=500
    ), batch_size=BATCH_SIZE
)


# Trainer 训练过程
# 包含训练迭代、检查训练期间测试误差以及保存所需要用来预测的模型参数。

# Event Handler 配置
# 可以在训练期间通过调用一个handler函数来监控培训进度
def event_handler(pass_id, batch_id, cost):
    print('Pass %d, Batch %d, Cost %f' % (pass_id, batch_id, cost))

# 开始训练
def train():
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    prediction, [avg_cost, acc] = train_program()

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    optimizer = optimizer_program()
    optimizer.minimize(avg_cost)

    PASS_NUM = 5
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]

    save_dirname = 'recognize_digits.inference.model'


    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_set = []
        avg_loss_set = []

        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(program=train_test_program,
                                          feed=train_test_feed.feed(test_data),
                                          fetch_list=[acc, avg_cost])
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))

        acc_val_mean = numpy.array(acc_set).mean()
        avg_loss_val_mean = numpy.array(avg_loss_set).mean()

        return avg_loss_val_mean, acc_val_mean

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)

    result_lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=[avg_cost, acc])
            if step % 100 == 0:
                event_handler(step, epoch_id, metrics[0])

            step += 1

        avg_cost_val, acc_val = train_test(train_test_program=test_program,
                                           train_test_reader=test_reader,
                                           train_test_feed=feeder)

        print('\nTest with Epoch %d, avg_cost: %s, acc: %s\n\n' % (epoch_id, avg_cost_val, acc_val))
        # Test with Epoch 4, avg_cost: 0.01788416613656345, acc: 0.9940286624203821

        result_lists.append((epoch_id, avg_cost_val, acc_val))

        if save_dirname is not None:
            fluid.io.save_inference_model(save_dirname, ['img'], [prediction], exe,
                                          model_filename=None,
                                          params_filename=None)

    best = sorted(result_lists, key=lambda list: float(list[1]))[0]

    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

    # prediction = softmax_regression()
    # Best pass is 4, testing Avgcost is 0.2824724394899265
    # The classification accuracy is 91.80%

    # prediction = multilayer_perception()
    # Best pass is 4, testing Avgcost is 0.05756924407188499
    # The classification accuracy is 98.16%

    # prediction = convolutional_n_n():
    # Best pass is 4, testing Avgcost is 0.01788416613656345
    # The classification accuracy is 99.40%
    # 从最简单的softmax回归变换到多层感知机，再到稍复杂的卷积神经网络的时候，
    # MNIST数据集上的识别准确率有了大幅度的提升，原因是卷积层具有局部连接和共享权重的特性。

# 应用模型:

def infer():

    save_dirname = 'recognize_digits.inference.model'

    # 生成预测输入数据
    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = cur_dir = os.getcwd()
    tensor_img = load_image(cur_dir + '/image/infer_3.png')

    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    infer_exe = fluid.Executor(place)

    infer_scope = fluid.core.Scope()
    with fluid.scope_guard(infer_scope):
        [infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(save_dirname, infer_exe)

        results = infer_exe.run(infer_program, feed={feed_target_names[0]: tensor_img},
                                fetch_list=fetch_targets)
        lab = numpy.argsort(results)
        print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])
        # Inference result of image/infer_3.png is: 3

if __name__ == '__main__':
    train()
    infer()
