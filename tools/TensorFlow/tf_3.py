
# 【AI实战】快速掌握TensorFlow（三）：激励函数
# https://my.oschina.net/u/876354/blog/1937296

# 常用的激励函数主要有：ReLU、ReLU6、sigmoid、tanh、softsign、ELU等。


# 2、怎样使用激励函数
# 位于神经网络库中（tensorflow.nn）

# （0）创建一个会话，调用默认计算图
import tensorflow as tf
sess = tf.Session()

# （1）ReLU函数
# ReLU（Rectifier linear unit，整流线性单元）是神经网络中最常用的激励函数
df = tf.nn.relu([-5., 0., 5., 10.])
print(sess.run(df))

# （2）ReLU6函数
# 引入ReLU6主要是为了抵消ReLU函数的线性增长部分，在ReLU的基础上再加上min
df = tf.nn.relu6([-5., 0., 5., 10.])
print(sess.run(df))

# （3）Leaky ReLU函数
# 引入Leaky ReLU主要是为了避免梯度消失，当神经元处于非激活状态时，允许一个非0的梯度存在，这样不会出现梯度消失，收敛速度快。
df = tf.nn.leaky_relu([-3., 0., 5.])
print(sess.run(df))

# （4）sigmoid函数
# sigmoid函数是神经网络中最常用的激励函数，它也被称为逻辑函数，它在深度学习的训练过程中会导致梯度消失，因此在深度学习中不怎么使用。
df = tf.nn.sigmoid([-1., 0., 1.])
print(sess.run(df))

# （5）tanh函数
# tanh函数即是双曲正切函数，tanh与sigmoid函数相似，但tanh的取值范围是0到1，sigmoid函数取值范围是-1到1。
df = tf.nn.tanh([-1., 0., 1.])
print(sess.run(df))

# （6）ELU函数
# ELU在正值区间的值为x本身，而在负值区间，ELU在输入取较小值时具有软饱和的特性，提升了对噪声的鲁棒性
df = tf.nn.elu([-1., 0., 1.])
print(sess.run(df))

# （7）softsign函数
# softsign函数是符号函数的连续估计
df = tf.nn.softsign([-1., 0., 1.])
print(sess.run(df))

# （8）softplus函数
# softplus是ReLU激励函数的平滑版
df = tf.nn.softplus([-1., 0., 1.])
print(sess.run(df))






