

# 在这个例子中，我们使用先前保存的CBOW词嵌入来改进我们对电影评论情绪的TF-IDF逻辑回归。

# 情感分析（Sentiment analysis）是一项非常艰巨的任务，因为人类语言使得很难掌握真正意义的微妙之处和细微差别。 讽刺，笑话，模糊的引用都使得任务成倍增加。


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
import text_helpers
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()




sess = tf.Session()

vocabulary_size = 2000
embedding_size = 50

batch_size = 100
max_words = 100

stops = stopwords.words('english')

print('loading data...')
texts, target = text_helpers.load_movie_data()

texts = text_helpers.normalize_text(texts, stops)

target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]  # 至少包含3个单词的句子
texts = [x for x in texts if len(x.split()) > 2]

train_indices = np.random.choice(len(target), round(len(target) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(target))) - set(train_indices)))
print(len(train_indices))  # 8325
print(len(train_indices) / len(texts))  # 0.8000192196809532

texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
# print("texts_train:", texts_train)
# texts_train: ['rock destined st centurys new conan hes going make splash even greater arnold schwarzenegger jeanclaud van damme steven segal',...]

target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

dict_file = os.path.join('.', 'temp', 'movie_vocab.pkl')
word_dictionary = pickle.load(open(dict_file, 'rb'))

text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))
print("text_data_train:", text_data_train)
# text_data_train: [list([527, 0, 0, 0, 33, 0, 214, 147, 16, 0, 7, 0, 1383, 0, 0, 1509, 0, 754, 0])
#  list([555, 0, 1764]) list([150, 4, 104, 17, 56, 0, 8, 211, 528]) ...
#  list([0, 204, 0, 233, 0, 596, 112, 0, 38, 494])
#  list([1222, 0, 0, 0, 841, 0, 42, 0, 0, 133, 984, 913, 202, 0, 1188, 0, 0, 0, 1576, 1804, 0])
#  list([79, 302, 4, 0, 0])]
print('text_data_train.shape:', text_data_train.shape)  # text_data_train.shape: (8325,)

# Pad/crop 到特定长度
text_data_train = np.array([x[0: max_words] for x in [y + [0] * max_words for y in text_data_train]])
text_data_test = np.array([x[0: max_words] for x in [y + [0] * max_words for y in text_data_test]])
print("text_data_train2222 :", text_data_train)
# text_data_train2222 : [[ 527    0    0 ...    0    0    0]
#  [ 555    0 1764 ...    0    0    0]
#  [ 150    4  104 ...    0    0    0]
#  ...
#  [   0  204    0 ...    0    0    0]
#  [1222    0    0 ...    0    0    0]
#  [  79  302    4 ...    0    0    0]]

print('text_data_train.shape2222:', text_data_train.shape)  # text_data_train.shape2222: (8325, 100)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  # shape: (2000, 50)

A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[None, max_words], dtype=tf.int32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

embed = tf.nn.embedding_lookup(embeddings, x_data)
print('embed.shape:', embed.shape)  # embed.shape: (?, 100, 50)
embed_avg = tf.reduce_mean(embed, 1)
print('embed_avg.shape:', embed_avg.shape)  # embed_avg.shape: (?, 50)

model_output = tf.add(tf.matmul(embed_avg, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.round(tf.sigmoid(model_output))
prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(prediction_correct)

my_opt = tf.train.AdagradOptimizer(0.005)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

model_ck_path = os.path.join('.', 'temp', 'cbow_movie_embeddings.ckpt')
saver = tf.train.Saver({'embeddings': embeddings})
saver.restore(sess, model_ck_path)
print('embeddings.shape:', embeddings.shape)  # embeddings.shape: (2000, 50)
print('embeddings:', sess.run(embeddings))  # embeddings: <tf.Variable 'Variable:0' shape=(2000, 50) dtype=float32_ref>


print('start training...')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []

for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size=batch_size)
    rand_x = text_data_train[rand_index]
    rand_y = np.transpose([target_train[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)
    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))
        # Generation # 10000. Train Loss (Test Loss): 0.70 (0.69). Train Acc (Test Acc): 0.37 (0.50)


# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()



# 预测不是很准确。 必须花更多的时间来调整CBOW嵌入。 我们也可以尝试深化我们的模型结构。