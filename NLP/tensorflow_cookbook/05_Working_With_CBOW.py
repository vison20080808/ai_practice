

# 在电影评论数据库上实现一种名为“CBOW”（连续词袋）的Word2Vec形式。 还介绍了保存和加载字嵌入的方法。

# 与skip-gram方法非常相似，只是我们预测来自环境词周围窗口的单个目标词。

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

import text_helpers


data_folder_name = 'temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)


sess = tf.Session()

batch_size = 200
vocabulary_size = 2000
embedding_size = 50
generations = 50000
model_learning_rate = 0.05

window_size = 3  # 考虑left-right单词

stops = stopwords.words('english')


print('Loading data...')
texts, target = text_helpers.load_movie_data()

texts = text_helpers.normalize_text(texts, stops)

target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]  # 至少包含3个单词的句子
texts = [x for x in texts if len(x.split()) > 2]


word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)

word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)


# 挑选五个测试单词
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
valid_examples = [word_dictionary[x] for x in valid_words]


print('Creating Model')


embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))  # stddev: 正态分布的标准差。

nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.zeros([batch_size, embedding_size])
for element in range(2 * window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])


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

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = optimizer.minimize(loss)

# 单词间的余弦相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


saver = tf.train.Saver({'embeddings': embeddings})

init = tf.global_variables_initializer()
sess.run(init)

text_data = [x for x in text_data if len(x) >= (2*window_size + 1)]


print('Training starting...')
loss_vec = []
loss_x_vec = []

for i in range(50000):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size, window_size, method='cbow')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    sess.run(train_step, feed_dict=feed_dict)

    if (i + 1) % 1000 == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print('Loss at step {} : {}'.format(i + 1, loss_val))
        # Loss at step 50000 : 3.9460363388061523

    if (i + 1) % 5000 == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5
            nearest = (-sim[j, :]).argsort()[1: top_k + 1]
            log_str = 'Nearest to {}: '.format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
            # Nearest to love:  including, compassion, grandeur, product, spielberg,
            # Nearest to hate:  unless, little, chilly, e, bold,
            # Nearest to happy:  weak, learn, gem, stealing, ugly,
            # Nearest to sad:  community, except, art, ambitious, sexual,
            # Nearest to man:  seems, sign, required, dead, apart,
            # Nearest to woman:  strong, pick, routine, rap, lines,

    if (i + 1) % 5000 == 0:
        with open(os.path.join(data_folder_name, 'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)

        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))
        # Model saved in file: ./ai_practice/NLP/tensorflow_cookbook/temp/cbow_movie_embeddings.ckpt

