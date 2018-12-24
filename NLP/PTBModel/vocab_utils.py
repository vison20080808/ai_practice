import codecs
import collections
from operator import itemgetter

RAW_DATA = './simple-examples/data/ptb.train.txt'
VOCAB_OUTPUT = 'ptb.vocab'  # 输出的词汇表

counter = collections.Counter()

with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(),
                            key=itemgetter(1),
                            reverse=True)

sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ['<eos>'] + sorted_words

# sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words  # 机器翻译数据时要用
# if len(sorted_words) > 10000:
#     sorted_words = sorted_words[0: 10000]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')



