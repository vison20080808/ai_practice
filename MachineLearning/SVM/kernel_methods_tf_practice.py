

# 《使用显式核方法改进线性模型》
# https://www.tensorflow.org/tutorials/representation/kernel_methods?hl=zh-cn

# 在本教程中，我们演示了结合使用（显式）核方法与线性模型可以如何大幅提高线性模型的预测质量，并且不会显著增加训练和推理时间。

# 目前，TensorFlow 仅支持密集特征的显式核映射

# 本教程包含以下步骤：
# 加载和准备 MNIST 数据，以用于分类。
# 构建一个简单的线性模型，训练该模型，并用评估数据对其进行评估。
# 将线性模型替换为核化线性模型，重新训练它，并重新进行评估。

import numpy as np
import tensorflow as tf

def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue = 3000):
    def _input_fn():
        images_batch, labels_batch = tf.train.shuffle_batch(
            tensors=[dataset_split.images, dataset_split.labels.astype(np.int32)],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True,
            num_threads=4)

        features_map = {'images': images_batch}
        return features_map, labels_batch
    return _input_fn

data = tf.contrib.learn.datasets.mnist.load_mnist()

train_input_fn = get_input_fn(data.train, batch_size=256)
eval_input_fn = get_input_fn(data.validation, batch_size=5000)


# 可以使用 MNIST 数据集训练一个线性模型。我们将使用 tf.contrib.learn.LinearClassifier Estimator，并用 10 个类别表示 10 个数字。输入特征会形成一个 784 维密集向量

import time
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=10)

start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()

print('Elapsed time: {} seconds'.format(end - start))
# Elapsed time: 79.61306405067444 seconds

eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
# {'loss': 0.25913033, 'accuracy': 0.9334, 'global_step': 2000}


# 除了调整（训练）批次大小和训练步数之外，您还可以微调一些其他参数。
# 例如，您可以更改用于最小化损失的优化方法，只需明确从可用优化器集合中选择其他优化器即可。
# 例如，以下代码构建的 LinearClassifier Estimator 使用了 Follow-The-Regularized-Leader (FTRL) 优化策略，并采用特定的学习速率和 L2 正则化。

optimizer = tf.train.FtrlOptimizer(learning_rate=5.0, l2_regularization_strength=1.0)
estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=10, optimizer=optimizer)

start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()

print('2222 Elapsed time: {} seconds'.format(end - start))
# 2222 Elapsed time: 89.15776586532593 seconds

eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print('2222:', eval_metrics)
# 2222: {'loss': 0.26833743, 'accuracy': 0.9252, 'global_step': 2000}

# 无论参数的值如何，线性模型可在此数据集上实现的准确率上限约为 93%。而且，会出现优化无效。


# 线性模型在 MNIST 数据集上的错误率相对较高（约 7%）表明输入数据不是可线性分隔的。我们将使用显式核映射减少分类错误。

# 直觉：大概的原理是，使用非线性映射将输入空间转换为其他特征空间（可能是更高维度的空间，其中转换的特征几乎是可线性分隔的），然后对映射的特征应用线性模型。

# 在本示例中，我们将使用 Rahimi 和 Recht 所著的论文“Random Features for Large-Scale Kernel Machines”（大型核机器的随机特征）中介绍的随机傅里叶特征来映射输入数据。
# 映射函数 RFFM，符合特性：  ≈ RBF（或高斯）核函数。
# 此函数是机器学习中使用最广泛的核函数之一，可隐式衡量比原始空间维度高得多的其他空间中的相似性。
# 要了解详情，请参阅径向基函数核 https://en.wikipedia.org/wiki/Radial_basis_function_kernel。

# tf.contrib.kernel_methods.KernelLinearClassifier 是预封装的 tf.contrib.learn Estimator，集显式核映射和线性模型的强大功能于一身。
# 其构造函数与 LinearClassifier Estimator 的构造函数几乎完全相同，但前者还可以指定要应用到分类器使用的每个特征的一系列显式核映射。
# 以下代码段演示了如何将 LinearClassifier 替换为 KernelLinearClassifier。


image_column = tf.contrib.layers.real_valued_column('images', dimension=784)

optimizer = tf.train.FtrlOptimizer(learning_rate=50.0, l2_regularization_strength=0.001)


# 以下：表示从 feature_columns 到 要应用到相应特征列的核映射列表的映射。

# 先使用随机傅里叶特征将初始的 784 维图像映射到 2000 维向量

# 请注意 stddev 参数。它是近似 RBF 核的标准偏差 (σ)，可以控制用于分类的相似性指标。stddev 通常通过微调超参数确定。
# 1.0 2.0 4.0 5.0 8.0 16.0

# 直观地来讲，映射的输出维度越大，两个映射向量的内积越逼近核，这通常意味着分类准确率越高。
# 换一种思路就是，输出维度等于线性模型的权重数；此维度越大，模型的“自由度”就越高。
# 不过，超过特定阈值后，输出维度的增加只能让准确率获得极少的提升，但却会导致训练时间更长。
kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=784, output_dim=2000, stddev=5.0, name='rffm')

kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(kernel_mappers=kernel_mappers, n_classes=10, optimizer=optimizer)


start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()

print('3333 Elapsed time: {} seconds'.format(end - start))
# 3333 Elapsed time: 122.72655177116394 seconds

eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print('3333:', eval_metrics)
# 3333: {'loss': 0.10176752, 'accuracy': 0.9716, 'global_step': 2000}

# kernel_mapper 中的 output_dim 改为：12000
# 3333 Elapsed time: 305.7560532093048 seconds
# 3333: {'loss': 0.07571213, 'accuracy': 0.9796, 'global_step': 2000}

# 显式核映射结合了非线性模型的预测能力和线性模型的可扩展性。与传统的双核方法不同，显式核方法可以扩展到数百万或数亿个样本。使用显式核映射时，请注意以下提示：
#
# 随机傅立叶特征对具有密集特征的数据集尤其有效。
# 核映射的参数通常取决于数据。模型质量与这些参数密切相关。通过微调超参数可找到最优值。
# 如果您有多个数值特征，不妨将它们合并成一个多维特征，然后向合并后的向量应用核映射。



