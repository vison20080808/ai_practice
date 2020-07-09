
# 【AI实战】快速掌握TensorFlow（四）：损失函数
# https://my.oschina.net/u/876354/blog/1940819

# 1、回归模型的损失函数
# 先定义预测结果（-1至1的等差序列）、目标结果（目标值为0）
import tensorflow as tf
sess = tf.Session()
y_pred = tf.linspace(-1., 1., 100)
y_target = tf.constant(0.)

# （1）L1正则损失函数（即绝对值损失函数）
loss_l1_vals = tf.abs(y_pred - y_target)
loss_l1_out = sess.run(loss_l1_vals)
print('l1:', loss_l1_out)

# （2）L2正则损失函数（即欧拉损失函数）
# L2正则损失函数是预测值与目标值差值的平方和
# 当对L2取平均值，就变成均方误差（MSE, mean squared error）
loss_l2_vals = tf.square(y_pred - y_target)
print('l2:', sess.run(loss_l2_vals))

loss_mse_vals = tf.reduce_mean(tf.square(y_pred - y_target))
print('mse:', sess.run(loss_mse_vals))

# L1正则损失函数在目标值附近不平滑，会导致模型不能很好地收敛。
# L2正则损失函数在目标值附近有很好的曲度，离目标越近收敛越慢，是非常有用的损失函数。

# （3）Pseudo-Huber 损失函数
# Huber损失函数经常用于回归问题，它是分段函数
# 公式可以看出当残差（预测值与目标值的差值，即y-f(x) ）很小的时候，损失函数为L2范数，残差大的时候，为L1范数的线性函数。
# Peseudo-Huber损失函数是Huber损失函数的连续、平滑估计，在目标附近连续
# delta=tf.constant(0.25)
# loss_huber_vals = tf.mul(tf.square(delta), tf.sqrt(1. + tf.square(y_target - y_pred)/delta)) - 1.)
# print('huber:', sess.run(loss_huber_vals))


# 2、分类模型的损失函数
# 预测值（-3至5的等差序列）和目标值（目标值为1）
y_pred = tf.linspace(-3., 5., 100)
y_target = tf.constant(1.)
y_targets = tf.fill([100, ], 1.)

# （1）Hinge损失函数
# Hinge损失常用于二分类问题。目标值为1，当预测值离1越近，则损失函数越小
loss_hinge_vals = tf.maximum(0., 1. - tf.multiply(y_target, y_pred))
print('hinge:', sess.run(loss_hinge_vals))

# （2）两类交叉熵（Cross-entropy）损失函数
# 当两个概率分布越接近时，它们的交叉熵也就越小
loss_ce_vals = tf.multiply(y_target, tf.log(y_pred)) - tf.multiply((1. - y_target), tf.log(1. - y_pred))
print('ce:', sess.run(loss_ce_vals))
# Cross-entropy损失函数主要应用在二分类问题上，预测值为概率值，取值范围为[0,1]

# （3）Sigmoid交叉熵损失函数
# 与上面的两类交叉熵类似，只是将预测值y_pred值通过sigmoid函数进行转换，再计算交叉熵损失
loss_sce_vals=tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_targets)
print('sce:', sess.run(loss_sce_vals))
# 由于sigmoid函数会将输入值变小很多，从而平滑了预测值，使得sigmoid交叉熵在预测值离目标值比较远时，其损失的增长没有那么的陡峭。

# （4）加权交叉熵损失函数
# 加权交叉熵损失函数是Sigmoid交叉熵损失函数的加权，是对正目标的加权。假定权重为0.5
weight = tf.constant(0.5)
loss_wce_vals = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_targets, pos_weight=weight)
print('wce:', sess.run(loss_sce_vals))

# （5）Softmax交叉熵损失函数
# Softmax交叉熵损失函数是作用于非归一化的输出结果，只针对单个目标分类计算损失。
# 通过softmax函数将输出结果转化成概率分布，从而便于输入到交叉熵里面进行计算（交叉熵要求输入为概率）
y_pred = tf.constant([[1., -3., 10.]])
y_target = tf.constant([[0.1, 0.02, 0.88]])
loss_softce_vals = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_target)
print('soft_ce:', sess.run(loss_softce_vals))


# 在实际使用中，对于回归问题经常会使用MSE均方误差（L2取平均）计算损失，对于分类问题经常会使用Sigmoid交叉熵损失函数。