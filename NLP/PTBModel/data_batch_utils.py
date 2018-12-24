import tensorflow as tf
import numpy as np

TRAIN_DATA = 'ptb.train'
TRAIN_BATCH_SIZE = 20  # batch 大小
TRAIN_NUM_STEP = 35  # 截断长度

def read_data(file_path):
    id_string = ''
    with open(file_path, 'r') as fin:
        for line in fin:
            id_string += line.strip() + '\n'
        # id_string = ' '.join([line.strip() for line in fin.readline()])

    # print(id_string)

    # 整个文档读进一个长字符串
    id_list = [int(w) for w in id_string.split()]

    return id_list


def make_batches(id_list, batch_size, num_step):

    # 每个batch包含的单词数量是 batch_size * num_step
    num_batches = (len(id_list) - 1) // (batch_size * num_step)
    print("make_batches() num_batches=", num_batches)

    # 将数据整理成一个维度为[batch_size, num_batches * num_step]的二维数组
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    data_batches = np.split(data, num_batches, axis=1)  # 按第二个维度，切分成num_batches个batch，存入一个数组
    # print(data_batches)

    label = np.array(id_list[1 : num_batches * batch_size * num_step + 1])  # 每个位置向右移1位，预测下一个单词
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)  # 按第二个维度，切分成num_batches个batch，存入一个数组
    # print(label_batches)

    return list(zip(data_batches, label_batches))



if __name__ == '__main__':
    id_list = read_data(TRAIN_DATA)
    # print(id_list)
    train_batches = make_batches(id_list, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    # print(train_batches)