import codecs
import sys


# RAW_DATA = './simple-examples/data/ptb.train.txt'
RAW_DATA = './simple-examples/data/ptb.valid.txt'
# RAW_DATA = './simple-examples/data/ptb.test.txt'

VOCAB = 'ptb.vocab'  # 词汇表
# OUTPUT_DATA = 'ptb.train'  # 将单词替换成为编号后的输出文件
OUTPUT_DATA = 'ptb.valid'  # 将单词替换成为编号后的输出文件
# OUTPUT_DATA = 'ptb.test'  # 将单词替换成为编号后的输出文件

vocab = []
with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    for line in f_vocab:
        vocab.append(line[:-1])
    # vocab = [w.strip() for w in f_vocab.readline()]

print(vocab)
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
print(word_to_id)


def get_id(word):
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['<unk>']


fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

for line in fin:
    words = line.strip().split() + ['<eos>']
    out_line = " ".join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)

fin.close()
fout.close()
