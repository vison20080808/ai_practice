
# NumPy数据分析练习
# https://www.numpy.org.cn/article/advanced/numpy_exercises_for_data_analysis.html


# NumPy数据分析问答
import numpy as np

print(np.__version__)

# 布尔数组
print(np.full((3, 3), True, dtype=bool))
print(np.ones((3, 3), dtype=bool))

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[arr % 2 == 1])  # 提取指定条件的元素

arr[arr % 2 == 1] = -1  # 替换指定条件的元素
print(arr)

# 不影响源
arr = np.arange(10)
out = np.where(arr % 2 == 1, -1, arr)
print(arr)
print(out)

# 改变数组形状
print(np.reshape(arr, (2, -1)))  # Setting to -1 automatically decides the number of cols


a = np.arange(10).reshape((2, -1))
b = np.repeat(1, 10).reshape(2, -1)

# 垂直叠加两个数组
c = np.concatenate([a, b], axis=0)
print(c)
print(np.vstack([a, b]))
print(np.r_[a, b])


# 水平叠加两个数组
c = np.concatenate([a, b], axis=1)
print(c)
print(np.hstack([a, b]))
print(np.c_[a, b])


# 10. 如何在无硬编码的情况下生成numpy中的自定义序列？
a = np.array([1, 2, 3])
print(np.r_[np.repeat(a, 3), np.tile(a, 3)])


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b))  # 公共项  # [2 4]
print(np.where(a == b))  # 公共项匹配位置  # (array([1, 3, 5, 7]),)

# 从数组a中删除数组b中的所有项。
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(np.setdiff1d(a, b))  # [1 2 3 4]


# 提取数字
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a >= 5) & (a <= 10))
print(a[index])

index = np.logical_and(a >= 5, a <= 10)
print(a[index])

print(a[(a >= 5) & (a <= 10)])


def maxx(x, y):
    if x >= y:
        return x
    else:
        return y

pair_max = np.vectorize(maxx, otypes=[float])

a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
print(pair_max(a, b))  # [6. 7. 9. 8. 9. 7. 5.]


# 交换二维数组的两列
arr = np.arange(9).reshape(3, 3)
print(arr)
print(arr[:, [1, 0, 2]])

# 交换二维数组的两行
print(arr[[1, 0, 2], :])

# 反转行
print(arr[::-1])

# 反转列
print(arr[:, ::-1])


# 创建包含5到10之间随机浮动的二维数组？
arr = np.arange(9).reshape(3, 3)
rand_arr = np.random.randint(low=5, high=10, size=(5, 3)) + np.random.random((5, 3))
print(rand_arr)

rand_arr = np.random.uniform(5, 10, size=(5, 3))
print(rand_arr)


# 只打印小数点后三位
rand_arr = np.random.random((5, 3))
np.set_printoptions(precision=3)
print(rand_arr[:4])

# e式科学记数法（如1e10）来打印
np.set_printoptions(suppress=False)
np.random.seed(100)
rand_arr = np.random.random([3, 3])/1e3
print(rand_arr)
np.set_printoptions(suppress=True, precision=6)
print(rand_arr)


# 打印的项数限制为最多6个元素
a = np.arange(15)
np.set_printoptions(threshold=6)
print(a)

# 打印完整的numpy数组而不截断
np.set_printoptions(threshold=np.nan)
print(a)


# 25. 如何导入数字和文本的数据集保持文本在numpy数组中完好无损？
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
print(iris[:3])

# 从前面问题中导入的一维鸢尾属植物数据集中提取文本列的物种。
iris_1d = np.genfromtxt(url, delimiter=',', dtype='object')
print(iris_1d.shape)
species = np.array([row[4] for row in iris_1d])
print(species[:5])


# 将1维元组数组转换为2维numpy数组？ 省略鸢尾属植物数据集种类的文本字段
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
print(iris_2d[:4])

iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
print(iris_2d[:4])


# 均值，中位数，标准差？第一列
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)

# 规范化数组，使数组的值正好介于0和1之间？
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin) / (Smax - Smin)
# or
S = (sepallength - Smin) / sepallength.ptp()  # Peak to peak (maximum - minimum) value along a given axis.
print(S)


# 计算Softmax得分？
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.array([float(row[0]) for row in iris])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

np.set_printoptions(precision=3)
print(softmax(sepallength))

# 第5和第95百分位数
print(np.percentile(sepallength, q=[5, 95]))


# 32. 如何在数组中的随机位置插入值？
# 在iris_2d数据集中的20个随机位置插入np.nan值
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

i, j = np.where(iris_2d)
# print(i, j)

np.random.seed(100)
iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan

print(iris_2d[:100])



# 33. 如何在numpy数组中找到缺失值的位置？
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print('Number of missing values: \n', np.isnan(iris_2d[:, 0]).sum())
print('Position of missing values: \n', np.where(np.isnan(iris_2d[:, 0])))

print(iris_2d.shape)  # (150, 4)
# print(iris_2d[:, 2])
# print(iris_2d[:, 0])
# 根据两个或多个条件过滤numpy数组？
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
print(iris_2d[condition])


# 35. 如何从numpy数组中删除包含缺失值的行？
any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
# print(any_nan_in_row)
print(iris_2d[any_nan_in_row][:5])

# 36. 如何找到numpy数组的两列之间的相关性？
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# Solution 1
print(np.corrcoef(iris[:, 0], iris[:, 2])[0, 1])

from scipy.stats.stats import pearsonr
corr, p_value = pearsonr(iris[:, 0], iris[:, 2])
print(corr, p_value)


# 37. 如何查找给定数组是否具有任何空值？
print(np.isnan(iris).any())


# 如何在numpy数组中用0替换所有缺失值？
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print(iris_2d[: 10])
iris_2d[np.isnan(iris_2d)] = 0
print(iris_2d[: 10])


# 39. 如何在numpy数组中查找唯一值的计数？
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

species = np.array([row.tolist()[4] for row in iris])
print(np.unique(species, return_counts=True))


# 40. 如何将数字转换为分类（文本）数组？
# 将iris_2d的花瓣长度（第3列）加入以形成文本数组，这样如果花瓣长度为：
#
# Less than 3 --> 'small'
# 3-5 --> 'medium'
# '>=5 --> 'large'

petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])  # Return the indices of the bins to which each value in input array belongs.
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

print(petal_length_cat[:10])


# 41. 如何从numpy数组的现有列创建新列？
# 在iris_2d中为卷创建一个新列，其中volume是（pi x petallength x sepal_length ^ 2）/ 3

iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')
volume = (np.pi * petallength * (sepallength ** 2)) / 3

volume = volume[:, np.newaxis]

out = np.hstack([iris_2d, volume])

print(out[:10])


# 42. 如何在numpy中进行概率抽样？
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Get the species column
species = iris[:, 4]

# Approach 1: Generate Probablistically
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
# print(species_out)

# Approach 2: Probablistic Sampling (preferred)
np.random.seed(100)
probs = np.r_[np.linspace(0, 0.5, num=50), np.linspace(0.501, .750, num=50),
    np.linspace(.751, 1.0, num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))


# 44. 根据sepallength列对虹膜数据集进行排序。
print(iris[iris[:, 0].argsort()][:50])


# 45. 如何在numpy数组中找到最常见的值？找到最常见的花瓣长度值（第3列）。
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])


# 46. 如何找到第一次出现的值大于给定值的位置？
# 第4列中查找第一次出现的值大于1.0的位置。
print(iris[:51])
print(iris[(np.argwhere(iris[:, 3].astype(float) > 1.0)[0])])


# 47. 如何将大于给定值的所有值替换为给定的截止值？
np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution 1: Using np.clip
print(np.clip(a, a_min=10, a_max=30))

# Solution 2: Using np.where
print(np.where(a < 10, 10, np.where(a > 30, 30, a)))


# 48. 如何从numpy数组中获取最大n值的位置？
# 获取给定数组a中前5个最大值的位置。
print(a)
print(a.argsort())
print(np.argpartition(-a, 5)[:5])

print(a[a.argsort()][-5:])
print(np.sort(a)[-5:])
print(np.partition(a, kth=-5)[-5:])
print(a[np.argpartition(-a, 5)][:5])


# 49. 如何计算数组中所有可能值的行数？
# 按行计算唯一值的计数。????
np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))
print(arr)


# 50. 如何将数组转换为平面一维数组？
# **给定：**
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)

arr_2d = np.array([a for arr in array_of_arrays for a in arr])
print(arr_2d)

arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)


# 51. 如何在numpy中为数组生成单热编码？
np.random.seed(101)
arr = np.random.randint(1,4, size=6)
print(arr)  # [2 3 2 2 2 1]

# print(arr[:, None] == np.unique(arr))
print((arr[:, None] == np.unique(arr)).view(np.int8))

def one_hot_encoding(arr):
    uniqs = np.unique(arr)
    print(uniqs)  # [1 2 3]
    out = np.zeros((arr.shape[0], uniqs.shape[0]))
    for i, k in enumerate(arr):
        print(i, k)
        out[i, k - 1] = 1

    return out

print(one_hot_encoding(arr))



# 54. 如何使用numpy对数组中的项进行排名？
np.random.seed(10)
a = np.random.randint(20, size=10)
print(a)

# print(a.argsort())
print(a.argsort().argsort())


# 55. 如何使用numpy对多维数组中的项进行排名？
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a, a.shape)

print(a.ravel().argsort().argsort().reshape(a.shape))


# 56. 如何在二维numpy数组的每一行中找到最大值？
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
print(a)

print(np.amax(a, axis=1))  # Return the maximum of an array or maximum along an axis.

print(np.apply_along_axis(np.max, arr=a, axis=1))

print(np.amin(a, axis=1))
print(np.apply_along_axis(lambda x: np.min(x) / np.max(x), arr=a, axis=1))


# 在给定的numpy数组中找到重复的条目(第二次出现以后)，并将它们标记为True。第一次出现应该是False的。
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print('Array: ', a)

out = np.full(a.shape[0], True)
unique_pos = np.unique(a, return_index=True)[1]
print(np.unique(a, return_index=True))

out[unique_pos] = False
print(out)

# 在二维数字数组中查找按分类列分组的数值列的平均值
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

numeric_column = iris[:, 1].astype('float')  # sepalwidth
grouping_column = iris[:, 4]  # species

output = [[group_val, numeric_column[grouping_column == group_val].mean()] for group_val in np.unique(grouping_column)]
print(output)

# 60. 如何将PIL图像转换为numpy数组？
from io import BytesIO
from PIL import Image
import PIL, requests

# URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
# response = requests.get(URL)
#
# I = Image.open(BytesIO(response.content))
#
# I = I.resize([150, 150])
# arr = np.asarray(I)
#
# im = PIL.Image.fromarray(np.uint8(arr))
# Image.Image.show(im)


# 从一维numpy数组中删除所有NaN值
a = np.array([1,2,3,np.nan,5,6,7,np.nan])
print(a)

print(a[~np.isnan(a)])


# 62. 如何计算两个数组之间的欧氏距离？
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])

dist = np.linalg.norm(a - b)
print(dist)


# 63. 如何在一维数组中找到所有的局部极大值(或峰值)？
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])

print(np.diff(a))  # [ 2  4 -6  1  4 -6  1]
print(np.sign(np.diff(a)))  # [ 1  1 -1  1  1 -1  1]
print(np.diff(np.sign(np.diff(a))))

doublediff = np.diff(np.sign(np.diff(a)))
peak_loc = np.where(doublediff == -2)[0] + 1
print(peak_loc)


# 从2d数组a_2d中减去一维数组b_1D，使得b_1D的每一项从a_2d的相应行中减去。
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3])

print(b_1d[:, None])
print(a_2d - b_1d[:, None])
# [[2 2 2]
#  [2 2 2]
#  [2 2 2]]


# 65. 如何查找数组中项的第n次重复索引？
# 找出x中数字1的第5次重复的索引。
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n = 5

print([i for i, v in enumerate(x) if v == 1][n - 1])
print(np.where(x == 1))  # (array([ 0,  2,  3,  7,  8, 10, 11]),)
print(np.where(x == 1)[0][n - 1])


# 66. 如何将numpy的datetime 64对象转换为datetime的datetime对象？
# **给定：** a numpy datetime64 object
dt64 = np.datetime64('2018-02-25 22:10:10')

print(dt64.tolist())

from datetime import datetime
print(dt64.astype(datetime))


# 67. 如何计算numpy数组的移动平均值？
# 对于给定的一维数组，计算窗口大小为3的移动平均值。

np.random.seed(100)
Z = np.random.randint(10, size=10)
print('array: ', Z)


def moving_average(a, n = 3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[: -n]
    return ret[n - 1:] / n


print(moving_average(Z, n=3).round(2))

print(np.convolve(Z, np.ones(3) / 3, mode='valid'))


# 68. 如何在给定起始点、长度和步骤的情况下创建一个numpy数组序列？
# 创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3。
def seq(start, length, step):
    end = start + (step * length)
    return np.arange(start, end, step)


length = 10
start = 5
step = 3
print(seq(start, length, step))  # [ 5  8 11 14 17 20 23 26 29 32]



# 69. 如何填写不规则系列的numpy日期中的缺失日期？
# 给定一系列不连续的日期序列。填写缺失的日期，使其成为连续的日期序列。
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)

for date, d in zip(dates, np.diff(dates)):
    print(date, d)

print([np.arange(date, (date + d)) for date, d in zip(dates, np.diff(dates))])
print(np.array([np.arange(date, (date + d)) for date, d in zip(dates, np.diff(dates))]))

filled_in = np.array([np.arange(date, (date + d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1)
output = np.hstack([filled_in, dates[-1]])
print(output)



# 70. 如何从给定的一维数组创建步长？
# 从给定的一维数组arr中，利用步进生成一个二维矩阵，窗口长度为4，步距为2，类似于 [[0,1,2,3], [2,3,4,5], [4,5,6,7]..]
def gen_strides(a, stride_len=2, window_len=4):
    n_strides = ((a.size - window_len) // stride_len) + 1  #(15 - 4) / 2 + 1 = 6
    print(n_strides)

    print(np.arange(0, n_strides * stride_len, stride_len))  # [ 0  2  4  6  8 10]

    return np.array([a[s:(s + window_len)] for s in np.arange(0, n_strides * stride_len, stride_len)])


print(gen_strides(np.arange(15)))
# [[ 0  1  2  3]
#  [ 2  3  4  5]
#  [ 4  5  6  7]
#  [ 6  7  8  9]
#  [ 8  9 10 11]
#  [10 11 12 13]]




