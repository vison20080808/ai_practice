

# TF-IDF：词频(Term Frequency) - 逆文本频率指数(Inverse Document Frequency)

# 02_bag_of_words中为句子中每个单词的出现赋予一个值。
# 这可能并不理想，因为每个类别的句子（先前食谱示例中的垃圾邮件和火腿）很可能具有相同频率的“the”，“and”和其他单词，
# 而像“viagra”和“sale”这样的单词可能应该 在确定文本是否是垃圾邮件方面，它的重要性日益增加。

# 每个条目中都会出现像“the”和“and”这样的词。
# 我们希望减轻这些词的重要性，因此我们可以想象将上述文本频率（TF）乘以整个文档频率的倒数可能有助于找到重要的单词。
# 但由于文本集（语料库）可能非常大，因此通常采用逆文档频率的对数。

# 创建TF-IDF向量要求我们将所有文本加载到内存中，并在开始训练模型之前计算每个单词的出现次数。因此，它没有在Tensorflow中完全实现，
# 我们将使用Scikit-learn来创建我们的TF-IDF嵌入，但使用Tensorflow来适应逻辑模型。

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()

batch_size = 200  #
max_features = 1000  # 将在逻辑回归中使用的tf-idf文本单词的最大数量。

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

target = [1. if x == 'spam' else 0. for x in target]

texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]


s = '''Good muffins cost $3.88\nin New York.  Please buy me two of them.\n\nThanks.'''
# nltk.download('punkt')  # 第一次启动nltk需要运行
print(nltk.word_tokenize(s))
# ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

def tokenizer(text):
    words = nltk.word_tokenize(text)
    # nltk.sent_tokenize(text) #按句子分割
    # nltk.word_tokenize(sentence) #分词
    return words


tf_idf = TfidfVectorizer(tokenizer=tokenizer, stop_words="english", max_features=max_features)
sparse_tf_idf_texts = tf_idf.fit_transform(texts)
print(sparse_tf_idf_texts)
#   (0, 632)	0.37172623140154337
#   (0, 170)	0.36805562944957004
#   (0, 48)	    0.3613966215413548
#   :	:
#   (5572, 457)	0.2553627074118946
#   (5572, 935)	0.3037138789526764
#   (5572, 200)	0.2986166134557511
#   (5572, 310)	0.4398216888575505
#   (5572, 96)	0.4520321847397513
#   (5572, 339)	0.3923502689151228
#   (5573, 870)	1.0


train_indices = np.random.choice(sparse_tf_idf_texts.shape[0], round(sparse_tf_idf_texts.shape[0] * 0.8), replace=False)
test_indices = np.array(list(set(range(sparse_tf_idf_texts.shape[0])) - set(train_indices)))
print(len(train_indices))  # 4459
print(len(train_indices) / len(texts))  # 0.7999641191245066

texts_train = sparse_tf_idf_texts[train_indices]
texts_test = sparse_tf_idf_texts[test_indices]

target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])



A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


model_output = tf.add(tf.matmul(x_data, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

#   x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
#   tf.round(x)  # [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
prediction = tf.round(tf.sigmoid(model_output))
prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(prediction_correct)

my_opt = tf.train.GradientDescentOptimizer(0.0025)  # 实现梯度下降算法的优化器 0.001
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []

for i in range(10000):
    # mini-batch gradient descent
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    if i == 0:
        # print('rand_y:', rand_y)
        print('texts_train[rand_index]:', texts_train[rand_index])
        # texts_train[rand_index]:   (0, 920)	1.0
        #   (1, 581)	0.20915268603890247
        #   (1, 397)	0.36944945981268684
        #   (1, 858)	0.5773208592481052
        #   (1, 965)	0.26226145623920327
        #   (1, 977)	0.3039218065584144

        print('type(texts_train[rand_index]):', type(texts_train[rand_index]))
        # type(texts_train[rand_index]): <class 'scipy.sparse.csr.csr_matrix'>

        print('rand_x:', rand_x)
        # rand_x: [[0. 0. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 0. 0. 0.]
        #  ...
        #  [0. 0. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 0. 0. 0.]]

        print('rand_x.shape:', rand_x.shape)  # rand_x.shape: (200, 1000)

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1)%100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy,
                                 feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)

    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))


# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

