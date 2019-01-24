

# http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/basics/recommender_system/index.html

# 推荐系统（Recommender System）

# 个性化推荐
# 个性化推荐系统是信息过滤系统（Information Filtering System）的子集，它可以用在很多领域，如电影、音乐、电商和 Feed 流推荐等。
# 推荐系统通过分析、挖掘用户行为，发现用户的个性化需求与兴趣特点，将用户可能感兴趣的信息或商品推荐给用户。
# 与搜索引擎不同，推荐系统不需要用户准确地描述出自己的需求，而是根据分析历史行为建模，主动提供满足用户兴趣和需求的信息。

# 传统的推荐系统方法主要有：
# 协同过滤推荐（Collaborative Filtering Recommendation）：计算一个用户与其他用户的相似度，利用目标用户的相似用户对商品评价的加权评价值，来预测目标用户对特定商品的喜好程度。
#                                                       存在冷启动的问题。同时也存在用户与商品之间的交互数据不够多造成的稀疏问题，会导致模型难以找到相近用户。
# 基于内容过滤推荐（Content-based Filtering Recommendation）：计算用户的兴趣和商品描述之间的相似度。
#                                                       简单直接，不需要依据其他用户对商品的评价。存在冷启动
# 组合推荐（Hybrid Recommendation）：运用不同的输入和技术共同进行推荐，以弥补各自推荐技术的缺点。

# 协同过滤是应用最广泛的技术之一，它又可以分为多个子类：
# 基于用户 （User-Based）的推荐[3] 、基于物品（Item-Based）的推荐[4]、基于社交网络关系（Social-Based）的推荐[5]、基于模型（Model-based）的推荐等。

# 基于用户：最古老；给用户推荐和他兴趣相似的其他用户喜欢的物品。关键是，计算用户的兴趣相似度。
# 基于物品：最广泛；给用户推荐和他之前喜欢的物品相似的物品。物品A和B具有很大的相似度，是因为喜欢A的用户大都也喜欢B。关键是，计算物品之间的相似度。
# 当用户数越来越大，计算用户兴趣相似度矩阵越来越难；而且基于用户 很难对推荐结果作出解释。

# 深度学习具有优秀的自动提取特征的能力，能够学习多层次的抽象特征表示，并对异质或跨域的内容信息进行学习，可以一定程度上处理推荐系统冷启动问题

# 首先介绍YouTube的视频推荐系统:
# 整个系统由两个神经网络组成：候选生成网络和排序网络。
# 候选生成网络从百万量级的视频库中生成上百个候选，排序网络对候选进行打分排序，输出排名最高的数十个结果。

# 候选生成网络（Candidate Generation Network）
# 候选生成网络将推荐问题建模为一个类别数极大的多类分类问题：

# 排序网络（Ranking Network）：结构类似于候选生成网络，但是它的目标是对候选进行更细致的打分排序。
# 不同之处是排序网络的顶部是一个加权逻辑回归（weighted logistic regression），它对所有候选视频进行打分，从高到底排序后将分数较高的一些视频返回给用户。




# 本节会使卷积神经网络（Convolutional Neural Networks）来学习电影名称的表示。
# 文本卷积神经网络（CNN）
# 首先，进行词向量的拼接操作：将每h个词拼接起来形成一个大小为h的词窗口

# 融合推荐模型：
# 在融合推荐模型的电影推荐系统中：使用用户特征和电影特征作为神经网络的输入，计算两个向量的余弦相似度，作为推荐系统的打分


# 数据准备:
# 使用MovieLens 百万数据集（ml-1m） http://files.grouplens.org/datasets/movielens/ml-1m.zip
# ml-1m 数据集包含了 6,000 位用户对 4,000 部电影的 1,000,000 条评价（评分范围 1~5 分，均为整数），由 GroupLens Research 实验室搜集整理。

import paddle

movie_info = paddle.dataset.movielens.movie_info()
print(movie_info.values())  # 电影特征
# dict_values([<MovieInfo id(1), title(Toy Story ), categories(['Animation', "Children's", 'Comedy'])>, ...）
# 电影的id是1，标题是《Toy Story》，该电影被分为到三个类别中。这三个类别是动画，儿童，喜剧。

user_info = paddle.dataset.movielens.user_info()
print(user_info.values())
# dict_values([<UserInfo id(1), gender(F), age(1), job(10)>, ...)
# 该用户ID是1，女性，年龄比18岁还年轻。职业ID是10。

# 对于每一条训练/测试数据，均为 <用户特征> + <电影特征> + 评分。
train_set_creator = paddle.dataset.movielens.train()
train_sample = next(train_set_creator())
print(train_sample)
# [1, 1, 0, 10, 1193, [15], [135, 2585, 5035, 4164, 1338, 2415], [5.0]]

uid = train_sample[0]

mov_id = train_sample[len(user_info[uid].value())]
print("User %s rates Movie %s with Score %s"%(user_info[uid], movie_info[mov_id], train_sample[-1]))
# User <UserInfo id(1), gender(F), age(1), job(10)> rates Movie <MovieInfo id(1193), title(One Flew Over the Cuckoo's Nest ), categories(['Drama'])> with Score [5.0]
# 用户1对电影1193的评价为5分


# 模型配置说明：
import math
import sys
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets

IS_SPARSE = True
USE_GPU = False
BATCH_SIZE = 256
PASS_NUM = 100  # 训练循环数

# 为用户特征综合模型定义模型配置
def get_usr_combined_features():
    # 对于每个用户，我们输入4维特征。其中包括user_id,gender_id,age_id,job_id。这几维特征均是简单的整数值。
    # 为了后续神经网络处理这些特征方便，我们借鉴NLP中的语言模型，将这几维离散的整数值，变换成embedding取出。
    # 分别形成usr_emb, usr_gender_emb, usr_age_emb, usr_job_emb。

    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1

    uid = layers.data(name='user_id', shape=[1], dtype='int64')

    usr_emb = layers.embedding(
        input=uid,
        dtype='float32',
        size=[USR_DICT_SIZE, 32],
        param_attr='user_table',
        is_sparse=IS_SPARSE)

    usr_fc = layers.fc(input=usr_emb, size=32)

    USR_GENDER_DICT_SIZE = 2

    usr_gender_id = layers.data(name='gender_id', shape=[1], dtype='int64')

    usr_gender_emb = layers.embedding(
        input=usr_gender_id,
        size=[USR_GENDER_DICT_SIZE, 16],
        param_attr='gender_table',
        is_sparse=IS_SPARSE)

    usr_gender_fc = layers.fc(input=usr_gender_emb, size=16)

    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = layers.data(name='age_id', shape=[1], dtype="int64")

    usr_age_emb = layers.embedding(
        input=usr_age_id,
        size=[USR_AGE_DICT_SIZE, 16],
        is_sparse=IS_SPARSE,
        param_attr='age_table')

    usr_age_fc = layers.fc(input=usr_age_emb, size=16)

    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")

    usr_job_emb = layers.embedding(
        input=usr_job_id,
        size=[USR_JOB_DICT_SIZE, 16],
        param_attr='job_table',
        is_sparse=IS_SPARSE)

    usr_job_fc = layers.fc(input=usr_job_emb, size=16)

    # 然后，我们对于所有的用户特征，均输入到一个全连接层(fc)中。将所有特征融合为一个200维度的特征。
    concat_embed = layers.concat(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)
    usr_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

    return usr_combined_features


# 对每一个电影特征做类似的变换，网络配置
def get_mov_combined_features():

    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1

    mov_id = layers.data(name='movie_id', shape=[1], dtype='int64')

    mov_emb = layers.embedding(
        input=mov_id,
        dtype='float32',
        size=[MOV_DICT_SIZE, 32],
        param_attr='movie_table',
        is_sparse=IS_SPARSE)

    mov_fc = layers.fc(input=mov_emb, size=32)

    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())

    category_id = layers.data(
        name='category_id', shape=[1], dtype='int64', lod_level=1)

    mov_categories_emb = layers.embedding(
        input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)

    mov_categories_hidden = layers.sequence_pool(
        input=mov_categories_emb, pool_type="sum")


    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())

    mov_title_id = layers.data(
        name='movie_title', shape=[1], dtype='int64', lod_level=1)

    mov_title_emb = layers.embedding(
        input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)

    # 电影标题名称(title)是一个序列的整数，整数代表的是这个词在索引序列中的下标。
    # 这个序列会被送入 sequence_conv_pool 层，这个层会在时间维度上使用卷积和池化。
    # 因为如此，所以输出会是固定长度，尽管输入的序列长度各不相同。
    mov_title_conv = nets.sequence_conv_pool(
        input=mov_title_emb,
        num_filters=32,
        filter_size=3,
        act="tanh",
        pool_type="sum")


    concat_embed = layers.concat(
        input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)

    mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

    return mov_combined_features


# 使用余弦相似度计算用户特征与电影特征的相似性。
def inference_program():
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
    scale_infer = layers.scale(x=inference, scale=5.0)

    label = layers.data(name='score', shape=[1], dtype='float32')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)

    return scale_infer, avg_cost


def optimizer_func():
    return fluid.optimizer.SGD(learning_rate=0.2)


def train(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.movielens.train(), buf_size=8192),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.movielens.test(), batch_size=BATCH_SIZE)

    feed_order = [
        'user_id', 'gender_id', 'age_id', 'job_id', 'movie_id', 'category_id',
        'movie_title', 'score'
    ]

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    scale_infer, avg_cost = inference_program()

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
        accumulated = len([avg_cost, scale_infer]) * [0]
        for test_data in reader():
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost, scale_infer])
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():
        feed_list = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list, place)
        exe.run(star_program)

        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                outs = exe.run(
                    program=main_program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost])
                out = np.array(outs[0])

                avg_cost_set = train_test(test_program, test_reader)

                test_avg_cost = np.array(avg_cost_set).mean()
                print("avg_cost: %s" % test_avg_cost)

                # if test_avg_cost < 4.0: # Change this number to adjust accuracy
                if batch_id == 5000:
                    if params_dirname is not None:
                        fluid.io.save_inference_model(params_dirname, [
                            "user_id", "gender_id", "age_id", "job_id",
                            "movie_id", "category_id", "movie_title"
                        ], [scale_infer], exe)
                    return
                else:
                    print('PassId {0}, BatchID {1}, Test Loss {2:0.2}'.format(
                        pass_id + 1, batch_id, float(test_avg_cost)))

                if math.isnan(float(out[0])):
                    sys.exit("got NaN loss, training failed.")

    train_loop()


# 在这个预测例子中，我们试着预测用户ID为1的用户对于电影'Hunchback of Notre Dame'的评分
def infer(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    infer_movie_id = 783
    infer_movie_name = paddle.dataset.movielens.movie_info()[
        infer_movie_id].title

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # 使用 create_lod_tensor(data, lod, place) 的API来生成细节层次的张量
        # data是一个序列，每个元素是一个索引号的序列。
        # lod是细节层次的信息，对应于data。
        # 比如，data = [[10, 2, 3], [2, 3]] 意味着它包含两个序列，长度分别是3和2。
        # 于是相应地 lod = [[3, 2]]，它表明其包含一层细节信息，意味着 data 有两个序列，长度分别是3和2。

        assert feed_target_names[0] == "user_id"
        user_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[1] == "gender_id"
        gender_id = fluid.create_lod_tensor([[np.int64(1)]], [[1]], place)

        assert feed_target_names[2] == "age_id"
        age_id = fluid.create_lod_tensor([[np.int64(0)]], [[1]], place)

        assert feed_target_names[3] == "job_id"
        job_id = fluid.create_lod_tensor([[np.int64(10)]], [[1]], place)

        assert feed_target_names[4] == "movie_id"
        movie_id = fluid.create_lod_tensor([[np.int64(783)]], [[1]], place)  # Hunchback of Notre Dame

        assert feed_target_names[5] == "category_id"
        category_id = fluid.create_lod_tensor(
            [np.array([10, 8, 9], dtype='int64')], [[3]], place)  # Animation, Children's, Musical

        assert feed_target_names[6] == "movie_title"
        movie_title = fluid.create_lod_tensor(
            [np.array([1069, 4140, 2923, 710, 988], dtype='int64')], [[5]],  # 'hunchback','of','notre','dame','the'
            place)

        results = exe.run(
            inferencer,
            feed={
                feed_target_names[0]: user_id,
                feed_target_names[1]: gender_id,
                feed_target_names[2]: age_id,
                feed_target_names[3]: job_id,
                feed_target_names[4]: movie_id,
                feed_target_names[5]: category_id,
                feed_target_names[6]: movie_title
            },
            fetch_list=fetch_targets,
            return_numpy=False)
        predict_rating = np.array(results[0])
        print("Predict Rating of user id 1 on movie \"" + infer_movie_name +
              "\" is " + str(predict_rating[0][0]))
        print("Actual Rating of user id 1 on movie \"" + infer_movie_name +
              "\" is 4.")


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "recommender_system.inference.model"
    train(use_cuda=use_cuda, params_dirname=params_dirname)
    infer(use_cuda=use_cuda, params_dirname=params_dirname)


if __name__ == '__main__':
    main(USE_GPU)

