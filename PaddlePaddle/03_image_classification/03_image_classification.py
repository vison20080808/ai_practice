

# http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/basics/image_classification/index.html

# 图像分类 是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。
# 图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

# 在深度学习算法之前使用较多的是基于词袋(Bag of Words)模型的物体分类方法。2012年之前的传统图像分类方法
# 对于图像而言，词袋方法需要构建字典。最简单的词袋模型框架可以设计为底层特征抽取、特征编码、分类器设计三个过程。

# 而基于深度学习的图像分类方法，可以通过有监督或无监督的方式学习层次化的特征描述，从而取代了手工设计或选择图像特征的工作。
# 深度学习模型中的卷积神经网络(Convolution Neural Network, CNN)近年来在图像领域取得了惊人的成绩

# 图像分类包括通用图像分类(cat-dog)、细粒度图像分类(花的种类)等。

# 图像识别领域大量的研究成果都是建立在PASCAL VOC、ImageNet等公开的数据集上，很多图像识别算法通常在这些数据集上进行测试和比较。
# PASCAL VOC是2005年发起的一个视觉挑战赛，ImageNet是2010年发起的大规模视觉识别竞赛(ILSVRC)的数据集

# Alex Krizhevsky在2012年ILSVRC提出的CNN模型取得了历史性的突破,获得了ILSVRC2012冠军，该模型被称作AlexNet。
# 目前的深度学习模型的识别能力已经超过了人眼。


# 传统CNN包含卷积层、全连接层等组件，并采用softmax多类别分类器和多类交叉熵损失函数
# 非线性变化: 在CNN里最常使用的为ReLu激活函数
# Dropout: 在模型训练阶段随机让一些隐层节点权重不工作，提高网络的泛化能力，一定程度上防止过拟合

# 在训练过程中由于每层参数不断更新，会导致下一次输入分布发生变化，这样导致训练过程需要精心设计超参数。
# 如2015年Sergey Ioffe和Christian Szegedy提出了Batch Normalization (BN)算法中，
# 每个batch对网络中的每一层特征都做归一化，使得每层分布相对稳定。
# BN算法不仅起到一定的正则作用，而且弱化了一些超参数的设计。
# 经过实验证明，BN算法加速了模型收敛过程，在后来较深的模型中被广泛使用。

# 接下来主要介绍VGG，GoogleNet和ResNet网络结构

# 牛津大学VGG(Visual Geometry Group)组在2014年ILSVRC提出的模型被称作VGG模型。基于ImageNet的VGG16模型

# GoogleNet 在2014年ILSVRC的获得了冠军，在介绍该模型之前我们先来了解NIN(Network in Network)模型 和Inception模块，
# 因为GoogleNet模型由多组Inception模块组成，模型设计借鉴了NIN的一些思想。
# GoogleNet-v4 引入下面要讲的ResNet设计思路。从v1到v4每一版的改进都会带来准确度的提升

# ResNet(Residual Network) 是2015年ImageNet图像分类、图像物体定位和图像物体检测比赛的冠军。
# ResNet提出了采用残差学习，在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。
# 50、101、152层网络连接示意图，使用的是瓶颈模块。这三个模型的区别在于每组中残差模块的重复次数不同(

# 数据准备：
# 通用图像分类公开的标准数据集常用的有CIFAR、ImageNet、COCO等，常用的细粒度图像分类数据集包括CUB-200-2011、Stanford Dog、Oxford-flowers等

# 常用的是ImageNet-2012数据集（规模相对较大），该数据集包含1000个类别：训练集包含1,281,167张图片，每个类别数据732至1300张不等，验证集包含50,000张图片，平均每个类别50张图片。
# CIFAR10数据集包含60,000张32x32的彩色图片，10个类别，每个类包含6,000张。其中50,000张图片作为训练集，10000张作为测试集。
# Paddle API提供了自动加载cifar数据集模块 paddle.dataset.cifar



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
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os


# Program Functions 配置：
# 提供了VGG和ResNet两个模型的配置

def vgg_bn_drop(input):
    # 一组卷积网络：groups决定每组VGG模块是几次连续的卷积操作，dropouts指定Dropout操作的概率。
    def conv_block(ipt, num_filter, groups, dropouts):

        # 由若干组 Conv->BN->ReLu->Dropout 和 一组 Pooling 组成
        # 卷积核大小为3x3，池化窗口大小为2x2，窗口滑动大小为2
        return fluid.nets.img_conv_group(input=ipt,
                                         pool_size=2,
                                         pool_stride=2,
                                         conv_num_filter=[num_filter] * groups,
                                         conv_filter_size=3,
                                         conv_act='relu',
                                         conv_with_batchnorm=True,
                                         conv_batchnorm_drop_rate=dropouts,
                                         pool_type='max')

    # 五组卷积操作。 第一、二组采用两次连续的卷积操作。第三、四、五组采用三次连续的卷积操作。每组最后一个卷积后面Dropout概率为0，即不使用Dropout操作。
    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    # 接两层512维的全连接
    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)

    # 分类器
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict


# ResNet模型的第1、3、4步和VGG模型相同
# 主要介绍第2步即CIFAR10数据集上ResNet核心模块
# resnet_cifar10：

# 带BN的卷积层
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=False):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)


# 残差模块的"直连"路径，"直连"实际分两种形式：
# 残差模块输入和输出特征通道数不等时，采用1x1卷积的升维操作；
# 残差模块输入和输出通道相等时，采用直连操作。
def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


# 一个基础残差模块，即图9左边所示，由两组3x3卷积组成的路径和一条"直连"路径组成。
def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


# 一个瓶颈残差模块，即图9右边所示，由上下1x1卷积和中间3x3卷积组成的路径和一条"直连"路径组成。
def bottleneck():  #【缺失？】
    pass


# 一组残差模块，由若干个残差模块堆积而成。每组中第一个残差模块滑动窗口大小与其他可以不同，以用来减少特征图在垂直和水平方向的大小。
def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp


# resnet_cifar10 的连接结构
def resnet_cifar10(ipt, depth=32):

    # 注意：除过第一层卷积层和最后一层全连接层之外，要求三组 layer_warp 总的含参层数能够被6整除，即 resnet_cifar10 的 depth 要满足 (depth−2) 。
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    nStages = {16, 64, 128}

    # 带BN的卷积层
    conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)

    # 连接3组残差模块
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)

    # 对网络做均值池化并返回该层
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)

    predict = fluid.layers.fc(input=pool, size=10, act='softmax')
    return predict


# Infererence Program 配置：

# 网络输入定义为 data_layer (数据层)，在图像分类中即为图像像素信息。
# CIFRAR10是RGB 3通道32x32大小的彩色图，因此输入数据大小为3072(3x32x32)
def inference_program():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    predict = resnet_cifar10(images, 32)
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict


# Train Program 配置
# 注意: 训练程序应该返回一个数组，第一个返回参数必须是 avg_cost。训练器使用它来计算梯度。
def train_program(prediction):
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return [avg_cost, acc]


# Optimizer Function 配置
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


# 训练模型
def train(use_cuda, params_dirname):
    # Data Feeders 配置
    BATCH_SIZE = 128
    print('start train() ...')
    print('cifar data downloading...')
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10(), buf_size=128 * 100),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)
    print('cifar data download finshed')

    feed_order = ['pixel', 'label']

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    predict = inference_program()
    avg_cost, acc = train_program(predict)

    # Test program
    test_program = main_program.clone(for_test=True)

    optimizer = optimizer_program()
    optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    EPOCH_NUM = 2

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost, acc]) * [0]
        for tid, test_data in enumerate(reader()):
            avg_cost_np = test_exe.run(program=program,
                                       feed=feeder_test.feed(test_data),
                                       fetch_list=[avg_cost, acc])
            accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]

            count += 1

        return [x_d / count for x_d in accumulated]

    def train_loop():
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]

        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)

        step = 0
        for pass_id in range(EPOCH_NUM):
            for step_id, data_train in enumerate(train_reader()):
                avg_cost_val, acc_val = exe.run(main_program,
                                         feed=feeder.feed(data_train),
                                         fetch_list=[avg_cost, acc])
                if step_id % 100 == 0:
                    print("\nPass %d, Batch %d, Cost %f, Acc %f" % (
                        step_id, pass_id, avg_cost_val, acc_val))
                    # Pass 300, Batch 1, Cost 0.927853, Acc 0.632812
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                step += 1

            avg_cost_test, accuracy_test = train_test(
                test_program, reader=test_reader)
            print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}\n'.format(
                pass_id, avg_cost_test, accuracy_test))
            # Test with Pass 1, Loss 1.4, Acc 0.54

            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["pixel"],
                                              [predict], exe)

    train_loop()


# 应用模型
def infer(use_cuda, params_dirname=None):
    from PIL import Image
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # 生成预测输入数据
    def load_image(infer_file):
        im = Image.open(infer_file)
        im = im.resize((32, 32), Image.ANTIALIAS)

        im = numpy.array(im).astype(numpy.float32)
        #加载图像的存储顺序为W（宽度）， H（高度），C（通道）。
        # PaddlePaddle需要CHW顺序，所以转置它们。
        im = im.transpose((2, 0, 1))  # CHW
        im = im / 255.0

        # Add one dimension to mimic the list format.
        im = numpy.expand_dims(im, axis=0)
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/image/dog.png')

    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # conv的输入维数应为4-D或5-D。
        # 使用inference_transpiler加速
        inference_transpiler_program = inference_program.clone()
        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(inference_transpiler_program, place)

        # 将Feed构造为{feed_target_name：feed_target_data}的字典
        # 结果将包含与fetch_targets对应的数据列表。
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: img},
                          fetch_list=fetch_targets)

        transpiler_results = exe.run(inference_transpiler_program,
                                     feed={feed_target_names[0]: img},
                                     fetch_list=fetch_targets)

        assert len(results[0]) == len(transpiler_results[0])
        for i in range(len(results[0])):
            numpy.testing.assert_almost_equal(
                results[0][i], transpiler_results[0][i], decimal=5)

        label_list = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
            "ship", "truck"
        ]

        print("infer results: %s" % label_list[numpy.argmax(results[0])])
        # infer results: dog


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "image_classification_resnet.inference.model"

    train(use_cuda=use_cuda, params_dirname=save_path)

    infer(use_cuda=use_cuda, params_dirname=save_path)


if __name__ == '__main__':
    main(use_cuda=False)