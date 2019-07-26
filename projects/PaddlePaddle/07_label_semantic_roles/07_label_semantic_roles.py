

# 语义角色标注（Semantic Role Labeling，SRL）

# 自然语言分析技术大致分为三个层面：词法分析、句法分析和语义分析。
# 语义角色标注是实现浅层语义分析的一种方式。

# 谓词是对主语的陈述或说明，指出“做什么”、“是什么”或“怎么样，代表了一个事件的核心，跟谓词搭配的名词称为论元。
# 语义角色是指论元在动词所指事件中担任的角色。
# 主要有：施事者（Agent）、受事者（Patient）、客体（Theme）、经验者（Experiencer）、受益者（Beneficiary）、工具（Instrument）、处所（Location）、目标（Goal）和来源（Source）等。

# ex.
# [小明]Agent[昨天]Time[晚上]Time在[公园]Location[遇到]Predicate了[小红]Patient。
# “遇到” 是谓词（Predicate，通常简写为“Pred”），“小明”是施事者（Agent），“小红”是受事者（Patient），“昨天” 是事件发生的时间（Time），“公园”是事情发生的地点（Location）。


# SRL 以句子的谓词为中心，不对句子所包含的语义信息进行深入分析，只分析句子中各成分与谓词之间的关系，即句子的谓词（Predicate）- 论元（Argument）结构，
# 并用语义角色来描述这些结构关系，是许多自然语言理解任务（如信息抽取，篇章分析，深度问答等）的一个重要中间步骤。
# 在研究中一般都假定谓词是给定的，所要做的就是找出给定谓词的各个论元和它们的语义角色。

# 传统的SRL系统大多建立在句法分析基础之上，通常包括5个流程：
# 1、构建一棵句法分析树
# 2、从句法树上识别出给定谓词的候选论元。
# 3、候选论元剪除；一个句子中的候选论元可能很多，候选论元剪除就是从大量的候选项中剪除那些最不可能成为论元的候选项。
# 4、论元识别：这个过程是从上一步剪除之后的候选中判断哪些是真正的论元，通常当做一个二分类问题来解决。
# 5、对第4步的结果，通过多分类得到论元的语义角色标签。

# 可以看到，句法分析是基础，并且后续步骤常常会构造的一些人工特征，这些特征往往也来自句法分析。


# “浅层句法分析”的思想应运而生。浅层句法分析也称为部分句法分析（partial parsing）或语块划分（chunking）

# 基于语块的SRL方法将SRL作为一个序列标注问题来解决。
# 序列标注任务一般都会采用「BIO表示方式」来定义序列标注的标签集，
# 在BIO表示法中，B代表语块的开始，I代表语块的中间，O代表语块结束。通过B、I、O 三种标记将不同的语块赋予不同的标签。
# 例如：对于一个由角色A拓展得到的语块组，将它所包含的第一个语块赋予标签B-A，将它所包含的其它语块赋予标签I-A，不属于任何论元的语块赋予标签O。

# 根据序列标注结果可以直接得到论元的语义角色标注结果，是一个相对简单的过程。这种简单性体现在：
# （1）依赖浅层句法分析，降低了句法分析的要求和难度；
# （2）没有了候选论元剪除这一步骤；
# （3）论元的识别和论元标注是同时实现的。
# 这种一体化处理论元识别和论元标注的方法，简化了流程，降低了错误累积的风险，往往能够取得更好的结果。



# 在本教程中我们也将SRL看作一个序列标注问题，不同的是，我们只依赖输入文本序列，不依赖任何额外的语法解析结果或是复杂的人造特征，利用深度神经网络构建一个端到端学习的SRL系统。
# 实践下面的任务：给定一句话和这句话里的一个谓词，通过序列标注的方式，从句子中找到谓词对应的论元，同时标注它们的语义角色。
# 我们以CoNLL-2004 and CoNLL-2005 Shared Tasks任务中SRL任务的公开数据集为例，


# 模型概览：
#
# 一、栈式循环神经网络（Stacked Recurrent Neural Network）
# 单层LSTM对状态转移的建模是 “浅” 的。堆叠多个LSTM单元，令前一个LSTMt时刻的输出，成为下一个LSTM单元t时刻的输入，称为第一个版本的栈式循环神经网络。
# 通常，堆叠4层LSTM单元可以正常训练，当层数达到4~8层时，会出现性能衰减，这时必须考虑一些新的结构以保证梯度纵向顺畅传播，这是训练深层LSTM网络必须解决的问题。

# 二、双向循环神经网络（Bidirectional Recurrent Neural Network）
# 可以看到历史，也看到未来
# 这种双向RNN结构和Bengio等人在机器翻译任务中使用的双向RNN结构

# 三、条件随机场 (Conditional Random Field)
# 在SRL任务中，深层LSTM网络学习输入的特征表示，条件随机场（Conditional Random Filed， CRF）在特征的基础上完成序列标注，处于整个网络的末端。
# CRF是一种概率化结构模型，可以看作是一个概率无向图模型，结点表示随机变量，边表示随机变量之间的概率依赖关系。
# 简单来讲，CRF学习条件概率P(X|Y)，其中 X=(x1,x2,...,xn) 是输入序列，Y=(y1,y2,...,yn) 是标记序列；
# 解码过程是给定 X序列求解令P(Y|X)最大的Y序列，即Y∗=arg maxYP(Y|X)。

# 序列标注任务只需要考虑输入和输出都是一个线性序列，并且由于我们只是将输入序列作为条件，不做任何条件独立假设，因此输入序列的元素之间并不存在图结构。
# 综上，在序列标注任务中使用的是如图5所示的定义在链式图上的CRF，称之为线性链条件随机场（Linear Chain Conditional Random Field）。

# 根据线性链条件随机场上的因子分解定理：p(Y|X)=1Z(X)exp(∑i=1n(∑jλjtj(yi−1,yi,X,i)+∑kμksk(yi,X,i)))
# ω是特征函数对应的权值，是CRF模型要学习的参数。
# 解码时，对于给定的输入序列X，通过解码算法（通常有：维特比算法、Beam Search）求令出条件概率P¯(Y|X)最大的输出序列 Y¯。

# 四、深度双向LSTM（DB-LSTM）SRL模型
# 在SRL任务中，输入是 “谓词” 和 “一句话”，目标是从这句话中找到谓词的论元，并标注论元的语义角色。如果一个句子含有n个谓词，这个句子会被处理n次。


# 引入：谓词上下文；谓词上下文区域标记

# 修改后的模型如下（图6是一个深度为4的模型结构示意图）：
#
# 1、构造输入
# 2、输入1是句子序列，输入2是谓词序列，输入3是谓词上下文，从句子中抽取这个谓词前后各n个词，构成谓词上下文，用one-hot方式表示，输入4是谓词上下文区域标记，标记了句子中每一个词是否在谓词上下文中；
# 3、将输入2~3均扩展为和输入1一样长的序列；
# 4、输入1~4均通过词表取词向量转换为实向量表示的词向量序列；其中输入1、3共享同一个词表，输入2和4各自独有词表；
# 5、第2步的4个词向量序列作为双向LSTM模型的输入；LSTM模型学习输入序列的特征表示，得到新的特性表示序列；
# 6、CRF以第3步中LSTM学习到的特征为输入，以标记序列为监督信号，完成序列标注；


# 数据介绍：
# 在此教程中，我们选用CoNLL 2005SRL任务开放出的数据集作为示例。
# 原始数据中同时包括了词性标注、命名实体识别、语法解析树等多种信息。
# 本教程中，我们使用test.wsj文件夹中的数据进行训练和测试，并只会用到words文件夹（文本序列）和props文件夹（标注结果）下的数据。
# 标注信息源自Penn TreeBank[7]和PropBank[8]的标注结果

# 原始数据需要进行数据预处理才能被PaddlePaddle处理，预处理包括下面几个步骤:
#
# 1、将文本序列和标记序列其合并到一条记录中；
# 2、一个句子如果含有n个谓词，这个句子会被处理n次，变成n条独立的训练样本，每个样本一个不同的谓词；
# 3、抽取谓词上下文和构造谓词上下文区域标记；
# 4、构造以BIO法表示的标记；
# 5、依据词典获取词对应的整数索引。
# 6、预处理完成之后一条训练样本包含9个特征，分别是：句子序列、谓词、谓词上下文（占 5 列，窗口=5）、谓词上下区域标志、标注序列。

# 除数据之外，我们同时提供了以下资源：
# word_dict	输入句子的词典，共计44068个词
# label_dict	标记的词典，共计106个标记
# predicate_dict	谓词的词典，共计3162个词
# emb	一个训练好的词表，32维

# 我们在英文维基百科上训练语言模型得到了一份词向量用来初始化SRL模型。在SRL模型训练过程中，词向量不再被更新。
# 我们训练语言模型的语料共有995,000,000个token，词典大小控制为4900,000词。CoNLL 2005训练语料中有5%的词不在这4900,000个词中，我们将它们全部看作未登录词，用<unk>表示。


#
from __future__ import print_function

import math, os
import numpy as np
import paddle
import paddle.dataset.conll05 as conll05
import paddle.fluid as fluid
import six
import time

with_gpu = os.getenv('WITH_GPU', '0') != '0'

# 获取词典，打印词典大小：
word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_dict_len = len(verb_dict)

print('word_dict_len: ', word_dict_len)  # word_dict_len:  44068
print('label_dict_len: ', label_dict_len)  # label_dict_len:  109
print('pred_dict_len: ', pred_dict_len)  # pred_dict_len:  3162


# 模型配置说明
mark_dict_len = 2   # 谓上下文区域标志的维度，是一个0-1 2值特征，因此维度为2
word_dim = 32       # 词向量维度
mark_dim = 5        # 谓词上下文区域通过词表被映射为一个实向量，这个是相邻的维度
hidden_dim = 512    # LSTM隐层向量的维度 ： 512 / 4
depth = 8           # 栈式LSTM的深度
mix_hidden_lr = 1e-3

IS_SPARSE = True
PASS_NUM = 10
BATCH_SIZE = 10

embedding_name = 'emb'


# 这里加载PaddlePaddle上版保存的二进制模型
def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


# 8个LSTM单元以“正向/反向”的顺序对所有输入序列进行学习。
def db_lstm(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark,
            **ignored):
    # 8 features
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        size=[pred_dict_len, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr='vemb')

    mark_embedding = fluid.layers.embedding(
        input=mark,
        size=[mark_dict_len, mark_dim],
        dtype='float32',
        is_sparse=IS_SPARSE)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    # Since word vector lookup table is pre-trained, we won't update it this time.
    # trainable being False prevents updating the lookup table during training.
    emb_layers = [
        fluid.layers.embedding(
            size=[word_dict_len, word_dim],
            input=x,
            param_attr=fluid.ParamAttr(
                name=embedding_name, trainable=False)) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    # 8 LSTM units are trained through alternating left-to-right / right-to-left order
    # denoted by the variable `reverse`.
    hidden_0_layers = [
        fluid.layers.fc(input=emb, size=hidden_dim, act='tanh')
        for emb in emb_layers
    ]

    hidden_0 = fluid.layers.sums(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    # In PaddlePaddle, state features and transition features of a CRF are implemented
    # by a fully connected layer and a CRF layer seperately. The fully connected layer
    # with linear activation learns the state features, here we use fluid.layers.sums
    # (fluid.layers.fc can be uesed as well), and the CRF layer in PaddlePaddle:
    # fluid.layers.linear_chain_crf only
    # learns the transition features, which is a cost layer and is the last layer of the network.
    # fluid.layers.linear_chain_crf outputs the log probability of true tag sequence
    # as the cost by given the input sequence and it requires the true tag sequence
    # as target in the learning process.

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    # 取最后一个栈式LSTM的输出和这个LSTM单元的输入到隐层映射，
    # 经过一个全连接层映射到标记字典的维度，来学习 CRF 的状态特征
    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out


# 训练模型：

def train(use_cuda, save_dirname=None, is_local=True):
    # define network topology

    # 句子序列
    word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)

    # 谓词
    predicate = fluid.layers.data(
        name='verb_data', shape=[1], dtype='int64', lod_level=1)

    # 谓词上下文5个特征
    ctx_n2 = fluid.layers.data(
        name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
    ctx_n1 = fluid.layers.data(
        name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
    ctx_0 = fluid.layers.data(
        name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
    ctx_p1 = fluid.layers.data(
        name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
    ctx_p2 = fluid.layers.data(
        name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)

    # 谓词上下区域标志
    mark = fluid.layers.data(
        name='mark_data', shape=[1], dtype='int64', lod_level=1)

    # define network topology
    feature_out = db_lstm(**locals())

    # 标注序列
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)

    # 学习 CRF 的转移特征
    crf_cost = fluid.layers.linear_chain_crf(
        input=feature_out,
        label=target,
        param_attr=fluid.ParamAttr(
            name='crfw', learning_rate=mix_hidden_lr))

    avg_cost = fluid.layers.mean(crf_cost)

    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=0.01,
            decay_steps=100000,
            decay_rate=0.5,
            staircase=True))

    sgd_optimizer.minimize(avg_cost)

    # The CRF decoding layer is used for evaluation and inference.
    # It shares weights with CRF layer.  The sharing of parameters among multiple layers
    # is specified by using the same parameter name in these layers. If true tag sequence
    # is provided in training process, `fluid.layers.crf_decoding` calculates labelling error
    # for each input token and sums the error over the entire sequence.
    # Otherwise, `fluid.layers.crf_decoding`  generates the labelling tags.
    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.conll05.test(), buf_size=8192),
        batch_size=BATCH_SIZE)
    # conll05.test()每次产生一条样本，包含9个特征，shuffle和组完batch后作为训练的输入。

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()


    feeder = fluid.DataFeeder(
        feed_list=[
            word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, predicate, mark, target
        ],
        place=place)
    exe = fluid.Executor(place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)

        start_time = time.time()
        batch_id = 0
        for pass_id in six.moves.xrange(PASS_NUM):
            for data in train_data():
                cost = exe.run(main_program,
                               feed=feeder.feed(data),
                               fetch_list=[avg_cost])
                cost = cost[0]

                if batch_id % 10 == 0:
                    print("avg_cost: " + str(cost))
                    # avg_cost: [35.201077]
                    if batch_id != 0:
                        print("second per batch: " + str((time.time(
                        ) - start_time) / batch_id))
                        # second per batch: 0.24249760456411557
                    # Set the threshold low to speed up the CI test
                    if float(cost) < 30.0:  # 60.0:
                        if save_dirname is not None:
                            fluid.io.save_inference_model(save_dirname, [
                                'word_data', 'verb_data', 'ctx_n2_data',
                                'ctx_n1_data', 'ctx_0_data', 'ctx_p1_data',
                                'ctx_p2_data', 'mark_data'
                            ], [feature_out], exe)
                        return

                batch_id = batch_id + 1

    train_loop(fluid.default_main_program())


# 应用模型

def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        # Setup inputs by creating LoDTensors to represent sequences of words.
        # Here each word is the basic element of these LoDTensors and the shape of
        # each word (base_shape) should be [1] since it is simply an index to
        # look up for the corresponding word vector.
        # Suppose the length_based level of detail (lod) info is set to [[3, 4, 2]],
        # which has only one lod level. Then the created LoDTensors will have only
        # one higher level structure (sequence of words, or sentence) than the basic
        # element (word). Hence the LoDTensor will hold data for three sentences of
        # length 3, 4 and 2, respectively.
        # Note that lod info should be a list of lists.
        lod = [[3, 4, 2]]
        base_shape = [1]
        # The range of random integers is [low, high]
        word = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        pred = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=pred_dict_len - 1)
        ctx_n2 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_n1 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_0 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_p1 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_p2 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        mark = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=mark_dict_len - 1)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        assert feed_target_names[0] == 'word_data'
        assert feed_target_names[1] == 'verb_data'
        assert feed_target_names[2] == 'ctx_n2_data'
        assert feed_target_names[3] == 'ctx_n1_data'
        assert feed_target_names[4] == 'ctx_0_data'
        assert feed_target_names[5] == 'ctx_p1_data'
        assert feed_target_names[6] == 'ctx_p2_data'
        assert feed_target_names[7] == 'mark_data'

        results = exe.run(inference_program,
                          feed={
                              feed_target_names[0]: word,
                              feed_target_names[1]: pred,
                              feed_target_names[2]: ctx_n2,
                              feed_target_names[3]: ctx_n1,
                              feed_target_names[4]: ctx_0,
                              feed_target_names[5]: ctx_p1,
                              feed_target_names[6]: ctx_p2,
                              feed_target_names[7]: mark
                          },
                          fetch_list=fetch_targets,
                          return_numpy=False)
        print(results[0].lod())
        # [[0, 3, 7, 9]]
        np_data = np.array(results[0])
        print("Inference Shape: ", np_data.shape)
        # Inference Shape:  (9, 109)

def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "label_semantic_roles.inference.model"

    train(use_cuda, save_dirname, is_local)
    infer(use_cuda, save_dirname)


main(use_cuda=False)













