
import tensorflow as tf

MAX_LEN = 50
SOS_ID = 1


def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)

    # 根据空格将单词编号切分并放到一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))

    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))

    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    # 每一项由4个张量组成：[0][0] 源句子； [0][1] 源句子的长度； [1][0] 目标句子；  [1][1] 目标句子的长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空的句子和长度过长的句子
    def FileterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_lable, trg_len)) = (src_tuple, trg_tuple)

        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))

        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(FileterLength)


    # 解码器的输入trg_input形式为 <sos> X Y Z
    # 解码器的目标输出trg_label形式为 X Y Z <eos>
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    dataset = dataset.shuffle(10000)

    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([]))  # 目标句子长度是单个数字
    )

    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


