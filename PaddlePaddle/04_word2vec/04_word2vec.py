

# http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/basics/word2vec/index.html

# 词向量：词的向量表征，也称为word embedding。
# 词向量是自然语言处理中常见的一个操作，是搜索引擎、广告系统、推荐系统等互联网服务背后常见的基础技术。
#
# 最自然的方式恐怕莫过于向量空间模型(vector space model)。 在这种方式里，每个词被表示成一个实数向量（one-hot vector），其长度为字典大小

# “母亲节”-“康乃馨”。这两个词对应的one-hot vectors之间的距离度量，无论是欧氏距离还是余弦相似度(cosine similarity)，由于其向量正交，都认为这两个词毫无相关性。

# 在机器学习领域里，各种“知识”被各种模型表示，词向量模型(word embedding model)就是其中的一类。
# 通过词向量模型可将一个 one-hot vector映射到一个维度更低的实数向量（embedding vector）

# 词向量模型可以是概率模型、共生矩阵(co-occurrence matrix)模型或神经元网络模型。

# 传统做法是统计一个词语的共生矩阵X。X是一个|V|×|V| 大小的矩阵，Xij表示在所有语料中，词汇表V(vocabulary)中第i个词和第j个词同时出现的词数
# 对X做矩阵分解（如奇异值分解，Singular Value Decomposition），得到的U即视为所有词的词向量：X=USVT


# 本文，基于神经网络训练词向量的细节，以及如何用PaddlePaddle训练一个词向量模型。

# 当词向量训练好后，我们可以用数据可视化算法t-SNE 画出词语特征在二维上的投影。

# 介绍三个训练词向量的模型：N-gram模型，CBOW模型和Skip-gram模型，它们的中心思想都是通过上下文得到一个词出现的概率。


# 语言模型
# 语言模型旨在为语句的联合概率函数P(w1,...,wT)建模, 其中wi表示句子中的第i个词。
# 语言模型的目标是，希望模型对有意义的句子赋予大概率，对没意义的句子赋予小概率。
# 这样的模型可以应用于很多领域，如机器翻译、语音识别、信息检索、词性标注、手写识别等，它们都希望能得到一个连续序列的概率。

# 以信息检索为例，当你在搜索“how long is a football bame”时（bame是一个医学名词），搜索引擎会提示你是否希望搜索"how long is a football game",
# 这是因为根据 语言模型 计算出“how long is a football bame”的概率很低，而与bame近似的，可能引起错误的词中，game会使该句生成的概率最大。

# 实际上通常用条件概率表示语言模型：P(w_1, ..., w_T) = \prod_{t=1}^TP(w_t | w_1, ... , w_{t-1})


# N-gram模型
# 在计算语言学中，n-gram是一种重要的文本表示方法，表示一个文本中连续的n个项。
# 一般用每个n-gram的历史n-1个词语组成的内容来预测第n个词。
# N-gram模型的优化目标则是最大化目标函数: \frac{1}{T}\sum_t f(w_t, w_{t-1}, ..., w_{t-n+1};\theta) + R(\theta)
# 其中，f(wt,wt−1,...,wt−n+1) 表示根据历史n-1个词得到当前词wt的条件概率。R(θ) 表示参数正则项。

# CBOW模型通过一个词的上下文（各N个词）预测当前词。具体来说，不考虑上下文的词语输入顺序，CBOW是用上下文词语的词向量的均值来预测当前词

# CBOW的好处是对上下文词语的分布在词向量上进行了平滑，去掉了噪声，因此在小数据集上很有效。
# 而Skip-gram的方法中，用一个词预测其上下文，得到了当前词上下文的很多样本，因此可用于更大的数据集。
# Skip-gram模型的具体做法是，将一个词的词向量映射到2n个词的词向量（2n表示当前输入词的前后各n个词），然后分别通过softmax得到这2n个词的分类损失值之和。



# 数据准备：
# 使用Penn Treebank （PTB）（经Tomas Mikolov预处理过的版本）数据集。PTB数据集较小，训练速度快
# train 42068句	；valid 3370句；test	3761句

# 本章训练的是5-gram模型，表示在PaddlePaddle训练时，每条数据的前4个词用来预测第5个词。
# PaddlePaddle提供了对应PTB数据集的python包paddle.dataset.imikolov

# 数据预处理:
# 把数据集中的每一句话前后加上开始符号<s>以及结束符号<e>。然后依据窗口大小（本教程中为5），从头到尾每次向右滑动窗口并生成一条数据。
# 如"I have a dream that one day" 一句提供了5条数据：
# <s> I have a dream
# I have a dream that
# have a dream that one
# a dream that one day
# dream that one day <e>
# 最后，每个输入会按其单词次在字典里的位置，转化成整数的索引序列，作为PaddlePaddle的输入。


from __future__ import print_function
import paddle as paddle
import paddle.fluid as fluid
import six
import numpy
import sys
import math

EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 100  # 更大的BATCH_SIZE将使得训练更快收敛，但也会消耗更多内存。
PASS_NUM = 100

use_cuda = False  # set to True if training with GPU

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)



# words：[first_word, second_word, third_word, forth_word, next_word]
def inference_program(words, is_sparse):
    # 定义我们的 N-gram 神经网络结构。这个结构在训练和预测中都会使用到。
    # 因为词向量比较稀疏，我们传入参数 is_sparse == True, 可以加速稀疏矩阵的更新。
    embed_first = fluid.layers.embedding(
        input=words[0],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=words[1],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=words[2],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.layers.embedding(
        input=words[3],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(input=concat_embed,
                              size=HIDDEN_SIZE,
                              act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word


def train_program(predict_word):
    # The declaration of 'next_word' must be after the invoking of inference_program,
    # or the data input order of train program would be [next_word, firstw, secondw,
    # thirdw, fourthw], which is not correct.
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.AdagradOptimizer(
        learning_rate=3e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(8e-4))


def train(if_use_cuda, params_dirname, is_sparse=True):
    place = fluid.CUDAPlace(0) if if_use_cuda else fluid.CPUPlace()

    # paddle.dataset.imikolov.train() 返回读取器
    # 在PaddlePaddle中，读取器是一个Python的函数，每次调用，会读取下一条数据。它是一个Python的generator。
    # paddle.batch 会读入一个读取器，然后输出一个批次化了的读取器。
    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    word_list = [first_word, second_word, third_word, forth_word, next_word]
    feed_order = ['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw']

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    predict_word = inference_program(word_list, is_sparse)
    avg_cost = train_program(predict_word)
    test_program = main_program.clone(for_test=True)

    sgd_optimizer = optimizer_func()
    sgd_optimizer.minimize(avg_cost)

    exe = fluid.Executor(place)


    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost]) * [0]
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost])
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():
        step = 0
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_cost_np = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[avg_cost])

                if step % 10 == 0:
                    outs = train_test(test_program, test_reader)

                    print("Step %d: Average Cost %f" % (step, outs[0]))

                    # 如果平均成本低于5.8，我们认为该模型足以停止。为了获得更好的模型，应该目标avg_cost低于3.5
                    if outs[0] < 5.2: #5.8:
                        if params_dirname is not None:
                            fluid.io.save_inference_model(params_dirname, [
                                'firstw', 'secondw', 'thirdw', 'fourthw'
                            ], [predict_word], exe)
                        return

                step += 1
                if math.isnan(float(avg_cost_np[0])):
                    sys.exit("got NaN loss, training failed.")

        raise AssertionError("Cost is too large {0:2.2}".format(avg_cost_np[0]))


    train_loop()


# 可以用我们训练过的模型，在得知之前的 N-gram 后，预测下一个词。
def infer(use_cuda, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        data1 = [[numpy.int64(211)]]  # 'among'
        data2 = [[numpy.int64(6)]]  # 'a'
        data3 = [[numpy.int64(96)]]  # 'group'
        data4 = [[numpy.int64(4)]]  # 'of'
        lod = [[numpy.int64(1)]]

        first_word = fluid.create_lod_tensor(data1, lod, place)
        second_word = fluid.create_lod_tensor(data2, lod, place)
        third_word = fluid.create_lod_tensor(data3, lod, place)
        fourth_word = fluid.create_lod_tensor(data4, lod, place)

        assert feed_target_names[0] == 'firstw'
        assert feed_target_names[1] == 'secondw'
        assert feed_target_names[2] == 'thirdw'
        assert feed_target_names[3] == 'fourthw'

        results = exe.run(
            inferencer,
            feed={
                feed_target_names[0]: first_word,
                feed_target_names[1]: second_word,
                feed_target_names[2]: third_word,
                feed_target_names[3]: fourth_word
            },
            fetch_list=fetch_targets,
            return_numpy=False)

        print(numpy.array(results[0]))
        most_possible_word_index = numpy.argmax(results[0])
        print(most_possible_word_index)
        print([
            key for key, value in six.iteritems(word_dict) if value == most_possible_word_index
        ][0])

        print(results[0].recursive_sequence_lengths())
        np_data = numpy.array(results[0])
        print("Inference Shape: ", np_data.shape)


def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    params_dirname = "word2vec.inference.model"

    train(
        if_use_cuda=use_cuda,
        params_dirname=params_dirname,
        is_sparse=is_sparse)

    infer(use_cuda=use_cuda, params_dirname=params_dirname)


if __name__ == '__main__':
    main(use_cuda=use_cuda, is_sparse=True)


# 在信息检索中，我们可以根据向量间的余弦夹角，来判断query和文档关键词这二者间的相关性。
# 在句法分析和语义分析中，训练好的词向量可以用来初始化模型，以得到更好的效果。
# 在文档分类中，有了词向量之后，可以用聚类的方法将文档中同义词进行分组，
# 也可以用 N-gram 来预测下一个词。