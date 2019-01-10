
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/07_Natural_Language_Processing/02_Working_with_Bag_of_Words/02_bag_of_words.py

# 在此示例中，我们将下载并预处理火腿/垃圾邮件文本数据。 然后，我们将使用单热编码来制作一组用于逻辑回归的单词功能集。
#
# 我们将使用这些单热矢量进行逻辑回归来预测文本是垃圾邮件还是火腿。

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

save_file_name = os.path.join('temp', 'temp_spam_data.csv')

if not os.path.exists('temp'):
    os.makedirs('temp')

if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')

    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]

    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]
print(len(texts))  # 5574
print(len(target))  # 5574

target = [1 if x == 'spam' else 0 for x in target]

texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]

# 为了确定填充/裁剪的良好句子长度，我们绘制文本长度的直方图（以单词表示）。
# text_lengths = [len(x.split()) for x in texts]
# text_lengths = [x for x in text_lengths if x < 50]
# plt.hist(text_lengths, bins=50)
# plt.title('Histogram of # of Words in Texts')
# plt.show()


# 我们将所有文本裁剪/填充为25个字长。 我们还会过滤掉至少3次不出现的单词。
sentence_size = 25  # 文本大小限制为25个单词。这是包含词语的常见做法，因为它限制了文本长度对预测的影响。
min_word_freq = 3

vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
vocab_processor.transform(texts)
transformed_texts = np.array([x for x in vocab_processor.transform(texts)])
# print(transformed_texts)
print(transformed_texts.shape)  # (5574, 25)
embedding_size = len(np.unique(transformed_texts))
print(embedding_size)  # 8220

train_indices = np.random.choice(len(texts), round(len(texts) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
print(len(train_indices))  # 4459
print(len(train_indices) / len(texts))  # 0.7999641191245066

texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]

target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

print(tf.ones(shape=[embedding_size]))  # Tensor("ones:0", shape=(8220,), dtype=float32)
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))  # 单位矩阵   diag返回具有给定对角线值的对角张量。
print(identity_mat)  # Tensor("Diag:0", shape=(8220, 8220), dtype=float32)

A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# 接下来，我们使用先前的单位矩阵创建文本字嵌入查找。
#
# 我们的逻辑回归将使用单词的计数作为输入。 通过对行中的嵌入输出求和来创建计数。

x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
print("x_embed=", x_embed)  # Tensor("embedding_lookup:0", shape=(25, 8220), dtype=float32)
x_col_sums = tf.reduce_sum(x_embed, 0)
print("x_col_sums=", x_col_sums)  # Tensor("Sum:0", shape=(8220,), dtype=float32)

x_col_sums_2D = tf.expand_dims(x_col_sums, 0)  # 在第axis位置增加一个维度；相对的：tf.squeeze() 从tensor中删除所有大小是1的维度
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)
print("x_col_sums_2D=", x_col_sums_2D)  # Tensor("ExpandDims:0", shape=(1, 8220), dtype=float32)
print("model_output=", model_output)  # Tensor("Add:0", shape=(1, 1), dtype=float32)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.sigmoid(model_output)
my_opt = tf.train.GradientDescentOptimizer(0.001)  # 实现梯度下降算法的优化器
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

print('Starting Training OVer {} sentences.'.format(len(texts_train)))

loss_vec = []
train_acc_all = []
train_acc_avg = []

for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]

    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if (ix + 1) % 10 == 0:
        print('Training #' + str(ix + 1) + ': loss = ' + str(temp_loss))

    # 保持过去50次观测精度的尾随平均值； 获得单一观察的预测
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # 0 or 1
    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# 测试集准确性
print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
test_acc_all = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[ix]]]

    if (ix + 1) % 50 == 0:
        print('Test Observation #' + str(ix + 1))

    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    test_acc_temp = target_test[ix] == np.round(temp_pred)
    test_acc_all.append(test_acc_temp)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))  # 所有测试数据样本的平均准确率

plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
plt.title('Avg Training Acc Over Past 50 Iterations')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.show()

















