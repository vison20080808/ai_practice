
# http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/basics/machine_translation/index.html
# 机器翻译（machine translation, MT）

# 传统机器翻译方法：多为基于规则的翻译系统，需要由语言学家编写两种语言之间的转换规则，再将这些规则录入计算机。

# 统计机器翻译（Statistical Machine Translation, SMT）：转化规则是由机器自动从大规模的语料中学习得到的，而非我们人主动提供规则。
# 它克服了基于规则的翻译系统所面临的知识获取瓶颈的问题，但仍然存在许多挑战：
# 1）人为设计许多特征（feature），但永远无法覆盖所有的语言现象；
# 2）难以利用全局的特征；
# 3）依赖于许多预处理环节，如词语对齐、分词或符号化（tokenization）、规则抽取、句法分析等，而每个环节的错误会逐步累积，对翻译的影响也越来越大。

# 将深度学习应用于机器翻译任务的方法大致分为两类：
# 1）仍以统计机器翻译系统为框架，只是利用神经网络来改进其中的关键模块，如语言模型、调序模型等（见图1的左半部分）；
# 2）不再以统计机器翻译系统为框架，而是直接用神经网络将源语言映射到目标语言，即端到端的神经网络机器翻译（End-to-End Neural Machine Translation, End-to-End NMT），简称为NMT模型。


# 本教程主要介绍NMT模型

# 效果展示：
# 输入：这些 是 希望 的 曙光 和 解脱 的 迹象 .
# 如果设定显示翻译结果的条数（即柱搜索算法的宽度）为3，生成的英语句子如下：
# 0 -5.36816   These are signs of hope and relief . <e>
# 1 -6.23177   These are the light of hope and relief . <e>
# 2 -7.7914  These are the light of hope and the relief of hope . <e>


# 模型概览：

# 一、双向循环神经网络（Bi-directional Recurrent Neural Network）
# 按时间步展开的双向循环神经网络


# 二、NMT模型中典型的编码器-解码器（Encoder-Decoder）框架
# 用于解决由一个任意长度的源序列到另一个任意长度的目标序列的变换问题。
# 编码和解码的过程通常都使用RNN实现。 （双向GRU）

# 三、柱搜索（beam search）算法
# 是一种启发式图搜索算法，用于在图或树中搜索有限集合中的最优扩展节点，通常用在解空间非常大的系统（如机器翻译、语音识别）中，原因是内存无法装下图或树中所有展开的解。
# 柱搜索算法使用广度优先策略建立搜索树


# 数据：

# 本教程使用WMT-14数据集中的bitexts(after selection)作为训练集，dev+test data作为测试集和生成集。

# 因为完整的数据集数据量较大，为了验证训练流程，PaddlePaddle接口paddle.dataset.wmt14中默认提供了一个经过预处理的较小规模的数据集。
#
# 该数据集有193319条训练数据，6003条测试数据，词典长度为30000。因为数据规模限制，使用该数据集训练出来的模型效果无法保证。



# 模型配置说明：


from __future__ import print_function
import contextlib

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as pd
from paddle.fluid.executor import Executor
from functools import partial
import os, sys
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
hidden_dim = 32
word_dim = 16
batch_size = 2
max_length = 8
topk_size = 50
beam_size = 2

is_sparse = True
decoder_size = hidden_dim
model_save_dir = "machine_translation.inference.model"


# 实现编码器框架：
def encoder():
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
    return encoder_out


# 实现训练模式下的解码器：
def train_decoder(context):
    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        pre_state = rnn.memory(init=context, need_reorder=True)
        current_state = pd.fc(
            input=[current_word, pre_state], size=decoder_size, act='tanh')

        current_score = pd.fc(
            input=current_state, size=target_dict_dim, act='softmax')
        rnn.update_memory(pre_state, current_state)
        rnn.output(current_score)

    return rnn()


# 实现推测模式下的解码器：
def decode(context, is_sparse):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    pd.array_write(init_ids, array=ids_array, i=counter)
    pd.array_write(init_scores, array=scores_array, i=counter)

    cond = pd.less_than(x=counter, y=array_len)

    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_state = pd.array_read(array=state_array, i=counter)
        pre_score = pd.array_read(array=scores_array, i=counter)

        # expand the lod of pre_state to be the same with pre_score
        pre_state_expanded = pd.sequence_expand(pre_state, pre_score)

        pre_ids_emb = pd.embedding(
            input=pre_ids,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=is_sparse)

        # use rnn unit to update rnn
        current_state = pd.fc(input=[pre_state_expanded, pre_ids_emb],
                              size=decoder_size,
                              act='tanh')
        current_state_with_lod = pd.lod_reset(x=current_state, y=pre_score)
        # use score to do beam search
        current_score = pd.fc(input=current_state_with_lod,
                              size=target_dict_dim,
                              act='softmax')
        topk_scores, topk_indices = pd.topk(current_score, k=beam_size)
        # calculate accumulated scores after topk to reduce computation cost
        accu_scores = pd.elementwise_add(
            x=pd.log(topk_scores), y=pd.reshape(pre_score, shape=[-1]), axis=0)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids,
            pre_score,
            topk_indices,
            accu_scores,
            beam_size,
            end_id=10,
            level=0)

        pd.increment(x=counter, value=1, in_place=True)

        # update the memories
        pd.array_write(current_state, array=state_array, i=counter)
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)

        # update the break condition: up to the max length or all candidates of
        # source sentences have ended.
        length_cond = pd.less_than(x=counter, y=array_len)
        finish_cond = pd.logical_not(pd.is_empty(x=selected_ids))
        pd.logical_and(x=length_cond, y=finish_cond, out=cond)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=10)

    return translation_ids, translation_scores


def train_program():
    context = encoder()
    rnn_out = train_decoder(context)
    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.Adagrad(
        learning_rate=1e-4,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.1))


def train(use_cuda):
    EPOCH_NUM = 1

    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = [
        'src_word_id', 'target_language_word', 'target_language_next_word'
    ]

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            if event.step % 10 == 0:
                print('pass_id=' + str(event.epoch) + ' batch=' + str(
                    event.step))
            elif event.step % 90000 == 0:
                trainer.save_params(model_save_dir)

        if isinstance(event, EndEpochEvent):
            trainer.save_params(model_save_dir)

    trainer = Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_func)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=feed_order)


def main(use_cuda):
    train(use_cuda)


if __name__ == '__main__':
    use_cuda = False  # set to True if training with GPU
    main(use_cuda)