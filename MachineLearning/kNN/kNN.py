from numpy import array, tile
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels

def classify0(inX, dataSet, labels, k):
    #计算距离
    # print(dataSet.shape)
    dataSetSize = dataSet.shape[0] #shape的返回值是一个元组，元组的长度就是数组的维数，即ndim。而元组中每个整数分别代表数组在其相应维度（/轴）上的大小。print shape后返回（2,3），说明这是一个2行3列的矩阵。

    tileInX = tile(inX, (dataSetSize, 1))#tile共有2个参数，A指待输入数组，reps则决定A重复的次数。整个函数用于重复数组A来构建新的数组。 #https://blog.csdn.net/xiahei_d/article/details/52749395
    # print('tileInX = ', tileInX)
    diffMat = tileInX - dataSet
    print(type(diffMat), diffMat.ndim, diffMat.shape)
    sqDiffMat = diffMat ** 2
    # print('sqDiffMat = ', sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1) #每一行向量相加 https://www.cnblogs.com/chamie/p/4847332.html
    # print('sqDistances = ', sqDistances)
    distances = sqDistances ** 0.5
    # print('distances = ', distances)

    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort() #返回的是数组值从小到大的索引值
    print(type(sortedDistIndicies))#, 'sortedDistIndicies = ', sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    print(classCount) #e.g. {'B': 2, 'A': 1}

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #按照第二个元素（1）逆序排列（从大到小， 频率最高）
    print(type(sortedClassCount), 'sortedClassCount = ', sortedClassCount)

    return sortedClassCount[0][0]


from numpy import zeros, shape

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))

    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)#每列的最小值 ()
    print('minVals = ', minVals, shape(minVals), minVals.ndim) #(3,) 一维数组
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]

    tile1 = tile(minVals, (m, 1))
    print('tile1 = ', tile1, shape(tile1), tile1.ndim) #(1000, 3) 二维数组
    normDataSet = dataSet - tile1
    print('normDataSet = ', normDataSet, shape(normDataSet), normDataSet.ndim) #(1000, 3) 二维数组
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    import os
    file2 = os.path.join(os.path.dirname(__file__) + '/datingTestSet2.txt')
    datingDataMat, datingLabels = file2matrix(file2)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]

    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classiferResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], 3)
        print('the predict result is {}, the real answer is {}'.format(classiferResult, datingLabels[i]))
        if classiferResult != datingLabels[i]:
            errorCount += 1.0
    print("The total error rate is ", errorCount/float(numTestVecs))





if __name__ == '__main__':
    group, labels = createDataSet()
    print('result = ', classify0([0, 0], group, labels, 3))

    print('=======dating test========')
    import os
    file2 = os.path.join(os.path.dirname(__file__) + '/datingTestSet2.txt')
    datingDataMat, datingLabels = file2matrix(file2)
    print(datingDataMat)
    # print(datingLabels)

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()


    datingClassTest()






