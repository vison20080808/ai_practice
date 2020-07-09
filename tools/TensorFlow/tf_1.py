

# 【AI实战】快速掌握Tensorflow（一）：基本操作
# https://my.oschina.net/u/876354/blog/1930175


# Tensorflow最主要的特点是使用数据流图（data flow graphs）进行数值计算，由节点（Nodes）和边（Edges）组成，
# 其中，节点（Nodes）表示数据操作，边（Edges）表示节点间相互通信的多维数组，
# 这种在边与边之间流动（flow）的数据也被称为张量（tensor），故而得名Tensorflow。

import tensorflow as tf

hello = tf.constant('hello tensorflow!')
sess = tf.Session()
print(sess.run(hello))


row_dim = 3
col_dim = 3

# （1）张量
# 张量是TensorFlow的主要数据结构，用于操作计算图。
# 一个张量（Tensor）可以简单地理解为任意维的数组，张量的秩表示其维度数量。张量的秩不同，名称也不相同。
# a、标量：维度为0的Tensor，也就是一个实数
# b、向量：维度为1的Tensor
# c、矩阵：维度为2的Tensor
# d、张量：维度达到及超过3的Tensor

# 创建张量有以下主要4种方法：
# a、创建固定张量
constant_ts = tf.constant([1, 2, 3, 4])
zero_ts = tf.zeros([row_dim, col_dim])
ones_ts = tf.ones([row_dim, col_dim])
filled_ts = tf.fill([row_dim, col_dim], 123)
# b、创建相似形状张量
zeros_like = tf.zeros_like(constant_ts)
ones_like = tf.ones_like(constant_ts)
# c、创建序列张量
linear_ts = tf.linspace(start=0.0, stop=2, num=6)
seq_ts = tf.range(start=4, limit=16, delta=4)
# d、随机张量
randunif_ts = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)  # 结果返回从minval（包含）到maxval（不包含）的均匀分布的随机数
randnorm_ts = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)  # 生成正态分布的随机数，其中mean表示平均值，stddev表示标准差

# （2）占位符和变量
my_var = tf.Variable(tf.zeros([row_dim, col_dim]))
init_op = tf.global_variables_initializer()  # 【注意】声明变量后需要进行初始化才能使用，最常使用以下函数一次性初始化所有变量

a = tf.placeholder(tf.float32, shape=[2,])
b = tf.placeholder(tf.float32, shape=[2,])
adder_node = a + b
print(sess.run(adder_node, feed_dict={a: [2, 4], b: [5.2, 8]}))

# （3）操作
# TensorFlow张量的加、减、乘、除、取模的基本操作是：add()、sub()、multiply()、div()、mod()。
mul_node = tf.multiply(a, b)
print(sess.run(mul_node, feed_dict={a: [2, 4], b: [5.2, 8]}))
# 其中，乘法、除法有比较特殊之处，如果是要对浮点数进行整除，则使用floordiv()；如果是要计算两个张量间的点积，则使用cross()。
