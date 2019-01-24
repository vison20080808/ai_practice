
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import math

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
STACKED_NUM = 3
BATCH_SIZE = 128
USE_GPU = False


# 栈式双向LSTM抽象出了高级特征并把其映射到和分类类别数同样大小的向量上

def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    assert stacked_num % 2 == 1

    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction


def inference_program(word_dict):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    dict_dim = len(word_dict)
    # net = convolution_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)
    net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM, STACKED_NUM)
    return net


def train_program(prediction):
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)



def train(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    print("Loading IMDB word dict....")
    word_dict = paddle.dataset.imdb.word_dict()

    print("Reading training data....")
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=BATCH_SIZE)

    print("Reading testing data....")
    test_reader = paddle.batch(
        paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

    feed_order = ['words', 'label']  # 用来定义每条产生的数据和paddle.layer.data之间的映射关系。比如，imdb.train产生的第一列的数据对应的是words这个特征。
    pass_num = 1

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    prediction = inference_program(word_dict)
    train_func_outputs = train_program(prediction)
    avg_cost = train_func_outputs[0]

    test_program = main_program.clone(for_test=True)

    # [avg_cost, accuracy] = train_program(prediction)
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
        accumulated = len(train_func_outputs) * [0]
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=train_func_outputs)
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():

        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)

        for epoch_id in range(pass_num):
            for step_id, data in enumerate(train_reader()):
                metrics = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[var.name for var in train_func_outputs])
                print("step: {0}, Metrics {1}".format(
                    step_id, list(map(np.array, metrics))))
                if (step_id + 1) % 10 == 0:
                    avg_cost_test, acc_test = train_test(test_program,
                                                         test_reader)
                    print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                        step_id, avg_cost_test, acc_test))
                    # Step 29, Test Loss 0.47, Acc 0.78

                    print("Step {0}, Epoch {1} Metrics {2}".format(
                        step_id, epoch_id, list(map(np.array, metrics))))

                if math.isnan(float(metrics[0])):
                    sys.exit("got NaN loss, training failed.")
            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["words"],
                                              prediction, exe)

    train_loop()


def infer(use_cuda, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        reviews_str = [
            'read the book forget the movie',
            'this is a great movie',
            'this is very bad'
        ]
        reviews = [c.split() for c in reviews_str]

        # 把评论中的每个词对应到word_dict中的id。如果词典中没有这个词，则设为unknown。
        UNK = word_dict['<unk>']
        lod = []
        for c in reviews:
            lod.append([np.int64(word_dict.get(words, UNK)) for words in c])

        base_shape = [[len(c) for c in lod]]

        # 用create_lod_tensor来创建细节层次的张量。
        tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

        assert feed_target_names[0] == "words"
        results = exe.run(
            inferencer,
            feed={feed_target_names[0]: tensor_words},
            fetch_list=fetch_targets,
            return_numpy=False)
        np_data = np.array(results[0])
        for i, r in enumerate(np_data):
            print("Predict probability of ", r[0], " to be positive and ", r[1],
                  " to be negative for review \'", reviews_str[i], "\'")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_stacked_lstm.inference.model"
    train(use_cuda, params_dirname)
    infer(use_cuda, params_dirname)


if __name__ == '__main__':
    use_cuda = False  # set to True if training with GPU
    main(use_cuda)














