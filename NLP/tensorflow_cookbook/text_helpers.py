
import string
import os
import urllib.request
import io
import tarfile
import collections
import numpy as np
import requests
import gzip

import os,sys

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dirname not in sys.path:
    sys.path.insert(0, dirname)



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

            if len(tuple_data) > 0:
                batch, labels = [list(x) for x in zip(*tuple_data)]

        elif method == 'cbow':
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
            # if index == 1:
            #     print('batch_and_labels:', batch_and_labels)
            # batch_and_labels: [([903, 0, 0], 2), ([2, 0, 0, 945], 903), ([2, 903, 0, 945, 808], 0), ([2, 903, 0, 945, 808, 0], 0),
            # ([903, 0, 0, 808, 0, 0], 945), ([0, 0, 945, 0, 0, 0], 808), ([0, 945, 808, 0, 0, 1040], 0), ([945, 808, 0, 0, 1040, 768], 0),
            #  ([808, 0, 0, 1040, 768, 0], 0), ([0, 0, 0, 768, 0, 272], 1040), ([0, 0, 1040, 0, 272, 0], 768), ([0, 1040, 768, 272, 0, 0], 0),
            #  ([1040, 768, 0, 0, 0, 0], 272), ([768, 0, 272, 0, 0, 10], 0), ([0, 272, 0, 0, 10, 1253], 0), ([272, 0, 0, 10, 1253, 361], 0),
            #  ([0, 0, 0, 1253, 361, 0], 10), ([0, 0, 10, 361, 0, 162], 1253), ([0, 10, 1253, 0, 162, 232], 361), ([10, 1253, 361, 162, 232, 0], 0),
            # ([1253, 361, 0, 232, 0], 162), ([361, 0, 162, 0], 232), ([0, 162, 232], 0)]

            # 只保留具有一致2 * window_size的窗口
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
            # if index == 1:
            #     print('batch_and_labels 222:', batch_and_labels)
                # batch_and_labels 222: [([2, 903, 0, 945, 808, 0], 0), ([903, 0, 0, 808, 0, 0], 945), ([0, 0, 945, 0, 0, 0], 808),
                # ([0, 945, 808, 0, 0, 1040], 0), ([945, 808, 0, 0, 1040, 768], 0), ([808, 0, 0, 1040, 768, 0], 0),
                # ([0, 0, 0, 768, 0, 272], 1040), ([0, 0, 1040, 0, 272, 0], 768), ([0, 1040, 768, 272, 0, 0], 0), ([1040, 768, 0, 0, 0, 0], 272),
                #  ([768, 0, 272, 0, 0, 10], 0), ([0, 272, 0, 0, 10, 1253], 0), ([272, 0, 0, 10, 1253, 361], 0), ([0, 0, 0, 1253, 361, 0], 10),
                # ([0, 0, 10, 361, 0, 162], 1253), ([0, 10, 1253, 0, 162, 232], 361), ([10, 1253, 361, 162, 232, 0], 0)]

            if len(batch_and_labels) > 0:
                batch, labels = [list(x) for x in zip(*batch_and_labels)]  # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式


        else:
            raise ValueError('Method {} not implemented yet'.format(method))

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


# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    # Check if files are already downloaded
    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open('temp_movie_review_temp.tar.gz', 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # Extract tar.gz file into temp folder
        tar = tarfile.open('temp_movie_review_temp.tar.gz', "r:gz")
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