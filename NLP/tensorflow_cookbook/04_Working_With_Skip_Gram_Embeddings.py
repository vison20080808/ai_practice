

# 之前，我们没有考虑与创建单词嵌入相关的单词顺序。

# 2013年初，谷歌发布论文，方法命名为"word2vec"。

# 基本思想是创建捕获单词关系方面的单词嵌入（word embeddings）。

# 为了找到这样的嵌入，我们将使用一个神经网络来预测输入单词的周围单词。（skip-gram）
# 还可以轻松地切换它并尝试：在给定一组周围单词的情况下预测目标单词。（CBOW）
# 两者都是word2vec过程的变体。
# 前者，从目标词预测周围词（上下文）的现有方法 称为：skip-gram model


# 本项目
# 采样时：按窗口中间word作为x，窗口内其它word作为label(y)。
# 训练：
#    tf.nn.nce_loss 是word2vec的skip-gram模型的负例采样方式的函数
#    输出词嵌入模型 embeddings  shape: (5000, 100)
# 验证validation：拿5个词来按照模型计算余弦相似度top5，输出看效果。

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string
import requests
import collections
import io
import tarfile
import gzip
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()


batch_size = 100
vocabulary_size = 5000
embedding_size = 100
generations = 100000

window_size = 2  # 考虑left-right单词


def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    # Check if files are already downloaded
    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open(os.path.join(save_folder_name, 'temp_movie_review_temp.tar.gz'), 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # Extract tar.gz file into temp folder
        tar = tarfile.open(os.path.join(save_folder_name, 'temp_movie_review_temp.tar.gz'), "r:gz")
        tar.extractall(path='temp')
        tar.close()

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding='latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]

    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data)

    return (texts, target)


texts, target = load_movie_data()


# Normalize text
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]

    # 去除 标点符号
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    return (texts)


# 停用词
# import nltk
# nltk.download('stopwords')  # 第一次使用，需要开启下载
stops = stopwords.words('english')

texts = normalize_text(texts, stops)

target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]  # 至少包含3个单词的句子
texts = [x for x in texts if len(x.split()) > 2]



def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s  in sentences]
    words = [x for sublist in split_sentences for x in sublist]

    count = [['RARE', -1]]  # 初始化 [word, word_count] ， -1: unknown
    # 取词汇表大小的出现最多的词
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print('count = ', count)
    # count =  [['RARE', -1], ('film', 1445), ('movie', 1263), ('one', 726), ('like', 721),...]

    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)

    print('word_dict=', word_dict)
    # word_dict= {'RARE': 0, 'film': 1, 'movie': 2, 'one': 3, 'like': 4, 'story': 5, 'much': 6, 'even': 7, 'good': 8, 'comedy': 9, 'time': 10, 'characters': 11,...}
    return word_dict


def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence.split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0

            sentence_data.append(word_ix)
        data.append(sentence_data)

    return data


word_dictionary = build_dictionary(texts, vocabulary_size)

word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)


# 挑选五个测试单词。 我们期待出现同义词
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']
valid_examples = [word_dictionary[x] for x in valid_words]


def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []

    index = 0
    while len(batch_data) < batch_size:
        index = index + 1
        rand_sentence = np.random.choice(sentences)

        # 生成要查看的连续窗口
        window_sequences = [rand_sentence[max(ix - window_size, 0): (ix + window_size + 1)] for ix, x in enumerate(rand_sentence)]
        # if index == 1:
        #     print('window_sequences:', window_sequences)
            # window_sequences: [[78, 410, 624], [78, 410, 624, 3592], [78, 410, 624, 3592, 1312],
            # [410, 624, 3592, 1312, 167], [624, 3592, 1312, 167, 191], [3592, 1312, 167, 191, 0],
            # [1312, 167, 191, 0, 0], [167, 191, 0, 0, 25], [191, 0, 0, 25, 138], [0, 0, 25, 138, 3546],
            # [0, 25, 138, 3546, 0], [25, 138, 3546, 0, 260], [138, 3546, 0, 260, 43], [3546, 0, 260, 43, 0],
            # [0, 260, 43, 0, 721], [260, 43, 0, 721], [43, 0, 721]]

        # 每个窗口的感兴趣元素是中间那个词
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]
        # if index == 1:
        #     print('label_indices:', label_indices)
            # label_indices: [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


        # 拉出每个窗口的中心词，并为每个窗口创建一个元组
        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            # if index == 1:
            #     print('batch_and_labels:', batch_and_labels)
                #  batch_and_labels: [(78, [410, 624]), (410, [78, 624, 3592]), (624, [78, 410, 3592, 1312]),
                # (3592, [410, 624, 1312, 167]), (1312, [624, 3592, 167, 191]), (167, [3592, 1312, 191, 0]),
                # (191, [1312, 167, 0, 0]), (0, [167, 191, 0, 25]), (0, [191, 0, 25, 138]), (25, [0, 0, 138, 3546]),
                # (138, [0, 25, 3546, 0]), (3546, [25, 138, 0, 260]), (0, [138, 3546, 260, 43]), (260, [3546, 0, 43, 0]),
                # (43, [0, 260, 0, 721]), (0, [260, 43, 721]), (721, [43, 0])]

            # (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
            # if index == 1:
            #     print('tuple_data:', tuple_data)
                # tuple_data: [(78, 410), (78, 624), (410, 78), (410, 624), (410, 3592), (624, 78), (624, 410), (624, 3592),
                # (624, 1312), (3592, 410), (3592, 624), (3592, 1312), (3592, 167), (1312, 624), (1312, 3592), (1312, 167),
                # (1312, 191), (167, 3592), (167, 1312), (167, 191), (167, 0), (191, 1312), (191, 167), (191, 0), (191, 0),
                #  (0, 167), (0, 191), (0, 0), (0, 25), (0, 191), (0, 0), (0, 25), (0, 138), (25, 0), (25, 0), (25, 138),
                # (25, 3546), (138, 0), (138, 25), (138, 3546), (138, 0), (3546, 25), (3546, 138), (3546, 0), (3546, 260),
                # (0, 138), (0, 3546), (0, 260), (0, 43), (260, 3546), (260, 0), (260, 43), (260, 0), (43, 0), (43, 260),
                # (43, 0), (43, 721), (0, 260), (0, 43), (0, 721), (721, 43), (721, 0)]

        elif method == 'cbow':  # 下一节再考虑
            batch_and_labels = []
            tuple_data = []
        else:
            raise ValueError('Method {} not implemented yet'.format(method))

        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
        # if index == 1:
        #     print('batch_data:', batch_data)
            # batch_data: [78, 78, 410, 410, 410, 624, 624, 624, 624, 3592, 3592, 3592, 3592, 1312, 1312, 1312, 1312, 167,
            #  167, 167, 167, 191, 191, 191, 191, 0, 0, 0, 0, 0, 0, 0, 0, 25, 25, 25, 25, 138, 138, 138, 138, 3546, 3546,
            # 3546, 3546, 0, 0, 0, 0, 260, 260, 260, 260, 43, 43, 43, 43, 0, 0, 0, 721, 721]
        # if index == 1:
        #     print('label_data:', label_data)
            # label_data: [410, 624, 78, 624, 3592, 78, 410, 3592, 1312, 410, 624, 1312, 167, 624, 3592, 167, 191, 3592, 1312,
            # 191, 0, 1312, 167, 0, 0, 167, 191, 0, 25, 191, 0, 25, 138, 0, 0, 138, 3546, 0, 25, 3546, 0, 25, 138, 0, 260, 138,
            #  3546, 260, 43, 3546, 0, 43, 0, 0, 260, 0, 721, 260, 43, 721, 43, 0]

    # print('all batch_data:', batch_data)
    # all batch_data: [78, 78, 410, 410, 410, 624, 624, 624, 624, 3592, 3592, 3592, 3592, 1312, 1312, 1312, 1312, 167, 167, 167,
    #  167, 191, 191, 191, 191, 0, 0, 0, 0, 0, 0, 0, 0, 25, 25, 25, 25, 138, 138, 138, 138, 3546, 3546, 3546, 3546, 0, 0, 0, 0,
    #  260, 260, 260, 260, 43, 43, 43, 43, 0, 0, 0, 721, 721, 1096, 1096, 1277, 1277, 1277, 85, 85, 85, 85, 383, 383, 383, 383,
    # 1225, 1225, 1225, 1225, 31, 31, 31, 31, 35, 35, 35, 35, 0, 0, 0, 0, 375, 375, 375, 375, 114, 114, 114, 114, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 791, 791, 791, 155, 155]

    # print('all label_data:', label_data)
    # all label_data: [410, 624, 78, 624, 3592, 78, 410, 3592, 1312, 410, 624, 1312, 167, 624, 3592, 167, 191, 3592, 1312, 191,
    #  0, 1312, 167, 0, 0, 167, 191, 0, 25, 191, 0, 25, 138, 0, 0, 138, 3546, 0, 25, 3546, 0, 25, 138, 0, 260, 138, 3546, 260,
    # 43, 3546, 0, 43, 0, 0, 260, 0, 721, 260, 43, 721, 43, 0, 1277, 85, 1096, 85, 383, 1096, 1277, 383, 1225, 1277, 85, 1225,
    # 31, 85, 383, 31, 35, 383, 1225, 35, 0, 1225, 31, 0, 375, 31, 35, 375, 114, 35, 0, 114, 0, 0, 375, 0, 0, 375, 114, 0, 0,
    # 114, 0, 0, 791, 0, 0, 791, 155, 0, 0, 155, 0, 791]

    # 剪裁
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    # print('np batch_data:', batch_data)
    # np batch_data: [  78   78  410  410  410  624  624  624  624 3592 3592 3592 3592 1312
    #  1312 1312 1312  167  167  167  167  191  191  191  191    0    0    0
    #     0    0    0    0    0   25   25   25   25  138  138  138  138 3546
    #  3546 3546 3546    0    0    0    0  260  260  260  260   43   43   43
    #    43    0    0    0  721  721 1096 1096 1277 1277 1277   85   85   85
    #    85  383  383  383  383 1225 1225 1225 1225   31   31   31   31   35
    #    35   35   35    0    0    0    0  375  375  375  375  114  114  114
    #   114    0]

    # print('np label_data:', label_data)
    # np label_data: [[ 410]
    #  [ 624]
    #  [  78]
    #  [ 624]
    #  [3592]
    #  [  78]
    #  [ 410]
    #  ...
    #  [ 375]
    #  [   0]
    #  [   0]
    #  [ 375]]

    return (batch_data, label_data)



embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))  # stddev: 正态分布的标准差。

nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# tf.nn.nce_loss 是word2vec的skip-gram模型的负例采样方式的函数
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=int(batch_size/2),
                                     num_classes=vocabulary_size))
# 假设nce_loss之前的输入数据是K维的，一共有N个类，那么
# weight.shape = (N, K)
# bias.shape = (N)
# inputs.shape = (batch_size, K)
# labels.shape = (batch_size, num_true)
# num_true : 实际的正样本个数
# num_sampled: 采样出多少个负样本
# num_classes = N
# sampled_values: 采样出的负样本，如果是None，就会用不同的sampler去采样。待会儿说sampler是什么。
# remove_accidental_hits: 如果采样时不小心采样到的负样本刚好是正样本，要不要干掉
# partition_strategy：对weights进行embedding_lookup时并行查表时的策略。TF的embeding_lookup是在CPU里实现的，这里需要考虑多线程查表时的锁的问题。

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
train_step = optimizer.minimize(loss)

# 单词间的余弦相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# vocabulary_size = 5000
# embedding_size = 100
print('embeddings.shape:', embeddings.shape)  # embeddings.shape: (5000, 100)
print('norm.shape:', norm.shape)  # norm.shape: (5000, 1)
print('normalized_embeddings.shape:', normalized_embeddings.shape)  # normalized_embeddings.shape: (5000, 100)
print('valid_embeddings.shape:', valid_embeddings.shape)  # valid_embeddings.shape: (5, 100)
print('similarity.shape:', similarity.shape)  # similarity.shape: (5, 5000)

init = tf.global_variables_initializer()
sess.run(init)

sim_init = sess.run(similarity)

loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    sess.run(train_step,  feed_dict=feed_dict)

    if (i + 1) % 500 == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))
        # Loss at step 100000 : 3.5881261825561523

    if (i+1) % 10000 == 0:
        sim = sess.run(similarity)  # similarity.shape: (5, 5000)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5
            nearest = (-sim[j, :]).argsort()[1: top_k + 1]  # 通过取相似矩阵的负数，反向排序
            log_str = 'Nearest to {}: '.format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                score = sim[j, nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
            # Nearest to cliche:  wilder, runofthemill, entertaining, serial, crush,
            # Nearest to love:  english, peak, served, rambling, composition,
            # Nearest to hate:  era, belt, walk, buffs, explored,
            # Nearest to silly:  pants, fearless, ages, magnetic, builds,
            # Nearest to sad:  strangers, denzel, benignis, winds, scattered,

    if (i + 1) % generations == 0:
        print('embeddings:', sess.run(embeddings))
        # embeddings: [[ 0.29708818 -0.04425869  0.12072687 ...  0.00421239  0.01268907
        #    0.12618354]
        #  [ 0.09860618 -0.42795566  0.8360113  ...  0.15279707 -0.4618066
        #    0.03203823]
        #  [ 0.71987736  0.4886699   0.30342656 ...  0.41562536 -0.4403617
        #   -0.14665207]
        #  ...
        #  [-0.21681696  0.48521522  0.51376593 ... -0.23012543 -0.47499615
        #   -0.8773737 ]
        #  [ 0.8731866   0.36347753  0.56524795 ...  0.569452    0.38541105
        #    1.0065207 ]
        #  [ 0.8587313  -0.6082079   1.2960986  ...  0.29391134 -0.2219832
        #   -0.26822904]]

        print('norm:', sess.run(norm))
        # norm: [[2.2339602]
        #  [4.3103924]
        #  [4.104874 ]
        #  ...
        #  [6.134557 ]
        #  [6.049039 ]
        #  [6.3017907]]

        print('normalized_embeddings:', sess.run(normalized_embeddings))
        # normalized_embeddings: [[ 0.13298723 -0.01981176  0.05404164 ...  0.00188561  0.00568008
        #    0.05648424]
        #  [ 0.02287638 -0.09928462  0.19395249 ...  0.03544853 -0.10713795
        #    0.00743279]
        #  [ 0.17537136  0.11904626  0.0739186  ...  0.10125167 -0.10727776
        #   -0.03572633]
        #  ...
        #  [-0.03534354  0.0790954   0.08374948 ... -0.03751297 -0.07742958
        #   -0.14302152]
        #  [ 0.14435129  0.06008847  0.09344426 ...  0.09413925  0.06371443
        #    0.1663935 ]
        #  [ 0.13626783 -0.0965135   0.20567147 ...  0.04663934 -0.03522541
        #   -0.04256394]]

        print('valid_embeddings:', sess.run(valid_embeddings))
        # valid_embeddings: [[ 2.27991372e-01 -7.97487199e-02  1.07722901e-01 -2.18882281e-02
        #    1.45562306e-01 -2.91489772e-02 -7.01406822e-02 -8.78382847e-02
        #   -1.36814892e-01  6.70138821e-02 -6.20533712e-02  9.31889266e-02
        #    3.05014066e-02  9.95063409e-02  6.07712120e-02  5.84472790e-02
        #   -9.26306751e-03  2.48853657e-02 -1.52355596e-01  1.24734059e-01
        #    1.36527821e-01 -1.93601191e-01 -3.93304937e-02 -3.04030292e-02
        #    1.58729646e-02 -1.73875138e-01  9.18051824e-02  3.47206295e-02
        #   -2.71264289e-04 -4.56824079e-02  1.32255912e-01  3.18917744e-02
        #    9.26405191e-02 -5.89869209e-02  2.94654332e-02 -1.36385813e-01
        #   -1.95540950e-01  5.23643866e-02  5.69663197e-02  2.39253398e-02
        #   -1.41162381e-01  3.40166837e-02 -4.55145203e-02 -3.17209661e-02
        #   -2.77459472e-02 -3.18767652e-02  3.20932046e-02 -8.44439045e-02
        #    1.25434687e-02 -1.40171483e-01  3.95047106e-02  1.36053279e-01
        #    3.42138708e-02 -1.69843525e-01  1.65449947e-01  4.38042432e-02
        #   -1.22495554e-03 -2.65671443e-02  4.26030420e-02  5.95680438e-02
        #   -1.13381937e-01 -5.14621697e-02 -1.20548941e-01 -2.74228714e-02
        #    2.07162291e-01 -1.65030852e-01 -5.58324009e-02  1.51582509e-01
        #   -8.83448496e-02 -8.05162042e-02 -3.07850894e-02  6.71217516e-02
        #   -8.92784223e-02 -7.71340877e-02 -9.24114138e-02 -1.34964332e-01
        #   -1.79529652e-01 -7.69223422e-02  4.53593880e-02  5.57468869e-02
        #    1.39220759e-01 -1.91162825e-01  5.13127558e-02 -2.83720363e-02
        #    6.26559183e-02 -5.51598631e-02  2.01784715e-01 -4.33571935e-02
        #   -1.06374145e-01 -1.65253207e-02  6.55383691e-02 -8.84655938e-02
        #    6.22833930e-02 -1.07871853e-01 -2.36430109e-01  6.25480190e-02
        #    3.10773123e-02 -1.64061427e-01  9.86867119e-04  1.14786904e-02]
        #  。。。。。。。。。。。。。。。。。。。。。。。
        #  [ 1.64579868e-01  8.69057849e-02 -1.56625398e-02  1.73754305e-01
        #    2.59180572e-02  1.41082024e-02 -6.70650825e-02  8.02863166e-02
        #   -1.37917632e-02 -1.86192364e-01  5.01982458e-02 -1.25181839e-01
        #    1.49125099e-01  2.55650096e-02  1.41691566e-01 -8.55891779e-02
        #    1.12373650e-01 -1.78653505e-02  3.58929299e-02  6.58027679e-02
        #   -1.91544622e-01  7.24094138e-02  3.95505540e-02 -1.50955349e-01
        #   -6.96683824e-02 -7.09397867e-02 -1.43862423e-02  8.80554244e-02
        #    1.16751321e-01  1.20028287e-01  7.72706047e-02  1.22073039e-01
        #   -1.18803931e-02 -1.00771055e-01  8.02631229e-02 -7.94538036e-02
        #    1.89142656e-02 -1.01297148e-01  7.36584142e-02 -7.56868348e-02
        #   -9.71549898e-02 -4.89236321e-03  1.39860466e-01  8.55028555e-02
        #   -1.99699197e-02  7.98144564e-02 -8.60386901e-03  5.45142554e-02
        #    3.92804556e-02 -1.41881943e-01 -2.00877264e-01 -5.81479669e-02
        #    9.37762484e-02 -1.43293709e-01 -8.96690264e-02 -1.24975257e-01
        #    3.81310806e-02 -5.82655929e-02  1.40505746e-01 -2.67442758e-03
        #    1.22470059e-03  1.68997228e-01  6.75376803e-02  1.86797187e-01
        #   -1.59161448e-01  1.10943161e-01 -4.71871987e-04 -2.14064673e-01
        #    9.27283242e-02  1.32652879e-01 -4.60091978e-02 -1.15261547e-01
        #   -1.88245103e-01  4.58775535e-02 -1.45818023e-02  1.18240625e-01
        #   -1.28209844e-01  2.65955850e-02 -1.03352740e-01 -1.22406743e-01
        #    4.99416664e-02 -3.22394073e-02  1.30680069e-01  1.05601430e-01
        #    2.85610054e-02 -1.31970957e-01  1.32412776e-01 -4.70151156e-02
        #    5.07805161e-02 -1.25895264e-02  6.86863065e-02  6.44355342e-02
        #   -5.99118806e-02  6.68090209e-02  1.00053750e-01 -3.66524123e-02
        #    5.73373996e-02  1.12500358e-02  1.27373755e-01  1.61816120e-01]]

        print('similarity:', sess.run(similarity))
        # similarity: [[0.24160938 0.2246516  0.16686362 ... 0.20110078 0.2531032  0.16171895]
        #  [0.3053579  0.15703662 0.21251358 ... 0.08477911 0.24894974 0.13915369]
        #  [0.30901954 0.13956746 0.19841547 ... 0.09195294 0.30120692 0.11578649]
        #  [0.35239527 0.2282705  0.06636532 ... 0.16111214 0.16511339 0.01362798]
        #  [0.34297004 0.11274121 0.20109579 ... 0.01180967 0.12026383 0.19180402]]







































