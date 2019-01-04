

import numpy as np


# https://www.numpy.org.cn/article/index.html

# https://www.numpy.org.cn/article/basics/understanding_numpy.html
print("=========part1===========")
my_array = np.array([1, 2, 3, 4, 5])
print(my_array, my_array.shape)  # [1 2 3 4 5] (5,)

my_new_array = np.zeros((5))
print(my_new_array)

my_random_array = np.random.random((5))
print(my_random_array)

my_2d_array = np.zeros((2, 3))
print(my_2d_array)

my_2d_array_new = np.ones((2, 3))
print(my_2d_array_new)

my_array = np.array([[4, 5], [6, 1]])
print(my_array[0][1], my_array.shape)  # 5 (2, 2)

# 提取第二列
my_array_column_2 = my_array[:, 1]
print(my_array_column_2)  # [5 1]


a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
sum = a + b
print(sum)

difference = a - b
print(difference)

product = a * b
print(product)

quotient = a / b
print(quotient)

print('以上为逐元素乘法， 以下执行矩阵乘法:')
matrix_product = a.dot(b)
print('matrix product = ', matrix_product)  # [[19. 22.][43. 50.]]


# https://www.numpy.org.cn/article/basics/an_introduction_to_scientific_python_numpy.html
print("=========part2===========")
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange(5)
d = np.linspace(0, 2*np.pi, 5)
print(a, b, c, d)

a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(type(a), type(b), a[2, 4])
print(a[0, 1:4])  # [12 13 14]
print(a[1:4, 0])  # [16 21 26]
print(a[::2, ::2])  # [[11 13 15] [21 23 25] [31 33 35]]
print(a[:, 1])  # [12 17 22 27 32]

print(type(a))  # <class 'numpy.ndarray'>
print(a.dtype)  # int64
print(a.size)  # 25
print(a.shape)  # (5, 5)
print(a.itemsize)  # 8  每个项占用的字节数
print(a.ndim)  # 2 数组的维数
print(a.nbytes)  # 200 数组中的所有数据消耗掉的字节数 25*8


# 基本操作符
a = np.arange(25)
a = a.reshape((5, 5))

b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5, 5))

print(a)
print(b)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(a < b)
print(a > b)
print(a.dot(b))


# 数组特殊运算符
a = np.arange(10)
print(a.sum())
print(a.min())  # 0
print(a.max())  # 9
print(a.cumsum()) # [ 0  1  3  6 10 15 21 28 36 45]


# 索引进阶
a = np.arange(0, 100, 10)  #[ 0 10 20 30 40 50 60 70 80 90]
indices = [1, 5, -1]
b = a[indices]
print(a)  # [ 0 10 20 30 40 50 60 70 80 90]
print(b)  # [10 50 90]


# 布尔屏蔽
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)  # [0.         0.12822827 ... 6.15495704 6.28318531]  50个元素
b = np.sin(a)
print(a, b)
plt.plot(a, b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()


# 缺省索引
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(a)  # [ 0 10 20 30 40 50 60 70 80 90]
print(b)  # [ 0 10 20 30 40]
print(c)  # [50 60 70 80 90]


# Where 函数
a = np.arange(0, 100, 10)
b = np.where(a < 50)
c = np.where(a >= 50)[0]
print(a)  # [ 0 10 20 30 40 50 60 70 80 90]
print(b)  # (array([0, 1, 2, 3, 4]),)
print(b[0])  # [0 1 2 3 4]
print(c)  # [5 6 7 8 9]



# https://www.numpy.org.cn/article/basics/python_numpy_tutorial.html
print("=========part3===========")

# Python中经典快速排序算法
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x  in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))  # [1, 1, 2, 3, 6, 8, 10]

s = 'hello'
print(s.capitalize())  # 'Hello'
print(s.upper())  # 'HELLO'
print(s.rjust(7))  # '  hello'
print(s.center(7))  # ' hello '
print(s.replace('l', '(ell)'))  # 'he(ell)(ell)o'
print('   world '.strip())  # 'world'


# 容器(Containers)
xs = [3, 1, 2]
print(xs, xs[2])  # [3, 1, 2] 2
print(xs[-1])  # 2
xs[2] = 'foo'
print(xs)  # [3, 1, 'foo']
xs.append('bar')
print(xs)  # [3, 1, 'foo', 'bar']
x = xs.pop()
print(x, xs)  # bar [3, 1, 'foo']


animals = ['cat', 'dog', 'monkey']
print(type(animals))  # <class 'list'>

for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))


d = {'cat': 'cute', 'dog': 'furry'}
print(type(d))  # <class 'dict'>
print(d.get('monkey', 'N/A'))

for key, value in d.items():
    print('%s is %s' % (key, value))


animals = {'cat', 'dog'}
print(type(animals))  # <class 'set'>
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))


t = (5, 6)
print(type(t))  # <class 'tuple'>



a = np.zeros((2, 2))
print(a)

b = np.ones((1, 2))
print(b)

c = np.full((2, 2), 7)
print(c)

d = np.eye(2)
print(d)

e = np.random.random((2, 2))
print(e)



x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))  # 逐元素乘法

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))

print(x.dot(v))
print(np.dot(x, v))

print(x.dot(y))
print(np.dot(x, y))


x = np.array([[1, 2], [3, 4]])
print(np.sum(x))  # 10
print(np.sum(x, axis=0))  # 按列column  [4 6]
print(np.sum(x, axis=1))  # 按行row   [3 7]

print(x)
print(x.T)  # 矩阵转置

v = np.array([1, 2, 3])
print(v)  # [1 2 3]
print(v.T)  # [1 2 3]


print('广播(Broadcasting):::::')
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)

print(x, v, y)
# 假设我们要向矩阵的每一行添加一个常数向量
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

# 以上等同于通过垂直堆叠多个 v 副本来形成矩阵 vv，然后执行元素的求和x 和 vv
vv = np.tile(v, (4, 1))
print(vv)
# [[1 0 1]
#  [1 0 1]
#  [1 0 1]
#  [1 0 1]]

y = x + vv
print(y)


# Numpy广播允许我们在不实际创建v的多个副本的情况下执行此计算
y = x + v  # 即使x具有形状(4，3)和v具有形状(3,)，但由于广播的关系，该行的工作方式就好像v实际上具有形状(4，3)，其中每一行都是v的副本，并且求和是按元素执行的。
print(y)

# 将两个数组一起广播遵循以下规则：
#
# 如果数组不具有相同的rank，则将较低等级数组的形状添加1，直到两个形状具有相同的长度。
# 如果两个数组在维度上具有相同的大小，或者如果其中一个数组在该维度中的大小为1，则称这两个数组在维度上是兼容的。
# 如果数组在所有维度上兼容，则可以一起广播。
# 广播之后，每个阵列的行为就好像它的形状等于两个输入数组的形状的元素最大值。
# 在一个数组的大小为1且另一个数组的大小大于1的任何维度中，第一个数组的行为就像沿着该维度复制一样

# 以下是广播的一些应用：
v = np.array([1, 2, 3])
w = np.array([4, 5])

print(np.reshape(v, (3, 1)))  # [[1] [2] [3]]
print(np.reshape(v, (3, 1)) * w)  # we first reshape v to be a column vector of shape (3, 1); then broadcast it against w to yield an output of shape (3, 2)
# [[ 4  5]
#  [ 8 10]
#  [12 15]]

x = np.array([[1,2,3], [4,5,6]])
# @@@@@@@@ Add a vector to each row of a matrix @@@@@@@@@@
print(x + v)  # x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3)
# [[2 4 6]
#  [5 7 9]]

print(x.T)  # shape (3, 2)
# [[1 4]
#  [2 5]
#  [3 6]]
print((x.T + w))  # shape (3, 2)
# [[ 5  9]
#  [ 6 10]
#  [ 7 11]]
# @@@@@@@@@ Add a vector to each column of a matrix @@@@@@@
print((x.T + w).T)  # shape (2, 3)
# [[ 5  6  7]
#  [ 9 10 11]]
# Another solution:
print(np.reshape(w, (2, 1)))
# [[4]
#  [5]]
print(x + np.reshape(w, (2, 1)))
# [[ 5  6  7]
#  [ 9 10 11]]

# Multiply a matrix by a constant:
print(x * 2)
# [[ 2  4  6]
#  [ 8 10 12]]


from scipy.misc import imread, imsave, imresize

img = imread('assets/cat.jpg')
print(img.dtype, img.shape)

img_tinted = img * [1, 0.95, 0.9]

img_tinted = imresize(img_tinted, (300, 300))

imsave('assets/cat_tinted.jpg', img_tinted)


# 函数 scipy.io.loadmat 和 scipy.io.savemat 允许你读取和写入MATLAB文件。你可以在这篇文档中学习相关操作。

# SciPy定义了一些用于计算点集之间距离的有用函数。
# 函数scipy.spatial.distance.pdist计算给定集合中所有点对之间的距离：
from scipy.spatial.distance import pdist, squareform
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

d = squareform(pdist(x, 'euclidean'))
print(d)
# 类似的函数（scipy.spatial.distance.cdist）计算两组点之间所有对之间的距离; 你可以在这篇文档中阅读它。


import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()



# 子图：使用subplot函数在同一个图中绘制不同的东西
plt.subplot(2, 1, 1)

plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()



from scipy.misc import imread, imresize

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))
plt.show()





print("=========part4===========")
# https://www.numpy.org.cn/article/basics/numpy_matrices_vectors.html  numpy中的矩阵和向量

A = np.array([[1,-1,2], [3,2,0]])
print(A)

v = np.array([[2],[1],[3]])  # 列向量
print(v)

v = np.transpose(np.array([[2, 1, 3]]))  # 列向量
print('v2 = ', v)



# 用numpy求解方程组 Ax = b
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
x = np.linalg.solve(A, b)
print(x)
# [[ 1.]
#  [-1.]
#  [ 2.]]


# 应用：多元线性回归  (Xt X) B = Xt y
# Xt = np.transpose(X)
# XtX = np.dot(Xt,X)
# Xty = np.dot(Xt,y)
# beta = np.linalg.solve(XtX,Xty)
#
# prediction = np.dot(x,beta)

