

import numpy as np

import matplotlib.pyplot as plt

import scipy.optimize as op

def plotData(X, y, show = False):
    index0 = list()
    index1 = list()

    j = 0
    for i in y:
        if i == 0:
            index0.append(j)
        else:
            index1.append(j)
        j = j + 1

    plt.scatter(X[index0, 0], X[index0, 1], marker='o')
    plt.scatter(X[index1, 0], X[index1, 1], marker='+')

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not Admitted'], loc = 'upper right')

    if show:
        plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def costFunction(theta, X, y, my_lambda):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = np.sum(np.dot((-1 * y).T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m

    return J

def gradient(theta, X, y, my_lambda):
    (m, n) = X.shape

    # print(theta)
    theta = theta.reshape((n, 1))
    # print(theta)

    grad = np.dot(X.T, sigmoid(np.dot(X, theta)) - y) / m

    # print('grad.shape = ', grad.shape)  # grad.shape =  (3, 1)
    # print('grad.flatten().shape = ', grad.flatten().shape)  # grad.flatten().shape =  (3,)

    return grad.flatten()


def plotDecisionBoundary(theta, X, y):
    figure = plotData(X[:, 1:], y)
    m, n = X.shape
    # Only need 2 points to define a line, so choose two endpoints
    if n <= 3:
        point1 = np.min(X[:, 1])
        point2 = np.max(X[:, 1])
        point = np.array([point1, point2])
        plot_y = -1 * (theta[0] + theta[1] * point) / theta[2]
        plt.plot(point, plot_y, '-')
        plt.legend(['Admitted', 'Not admitted', 'Boundary'], loc='lower left')
    plt.show()
    return 0

def predict(theta, X):
    m, n = X.shape

    p = np.zeros((m, 1))
    k = np.where(sigmoid(X.dot(theta)) >= 0.5)

    p[k] = 1
    return p


def gradDescent(theta, X, y, alpha, iterations):
    m = len(y)
    # temp = theta

    for k in range(iterations):
        h = sigmoid(np.dot(X, theta))

        # for i in range(X.shape[1]):
        #     temp[i, 0] = theta[i, 0] - alpha / m * (np.sum((h - y) * X[:, i].reshape(m, 1)))
        # theta = temp

        grad = np.dot(X.T, h - y) / m
        theta = theta - alpha * grad

        # print('gradDescent() k = ', k, '; theta = ', theta, '; cost = ', costFunction(theta, X, y, None))

    return theta

if __name__ == '__main__':
    ex2_file = './ex2data1.txt'
    data = np.loadtxt(ex2_file, delimiter=',')
    print(np.shape(data))  # (100, 3)
    print(type(data))

    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    y = np.reshape(y, (y.shape[0], 1))  # 转换为列向量

    print(np.shape(X))  # (100, 2)
    print(np.shape(y))  # (100, 1)

    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples')
    plotData(X, y)

    print('=' * 40)

    (m, n) = X.shape
    X = np.column_stack((np.ones((m, 1)), X))
    (m, n) = X.shape

    initial_theta = np.zeros((n, 1))
    my_lambda = 0

    cost = costFunction(initial_theta, X, y, my_lambda)
    grad = gradient(initial_theta, X, y, my_lambda)
    print('Cost at initial theta (zeros):', cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): ')
    print(grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([[-24], [0.2], [0.2]])
    cost = costFunction(test_theta, X, y, my_lambda)
    grad = gradient(test_theta, X, y, my_lambda)
    print('Cost at initial theta :', cost)
    print('Expected cost (approx): 0.218\n')
    print('Gradient at initial theta : ')
    print(grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')
    print("=" * 40)

    myLambda = 1
    # Result = op.minimize(fun=costFunction, x0=initial_theta, args=(X, y, myLambda), method='TNC', jac=gradient)
    Result = op.minimize(fun=costFunction, x0=initial_theta,
                         args=(X, y, myLambda), method='TNC', jac=gradient)
    theta = Result.x
    cost = Result.fun
    print('Cost at theta found by fminunc:', cost)
    print('Expected cost (approx):0.203\n')

    # iterations = 400000
    # theta = gradDescent(initial_theta, X, y, 0.0004, iterations)

    print('theta:', theta)
    print('Expected theta (approx):[-25.161  0.206  0.201]\n')
    plotDecisionBoundary(theta, X, y)
    print('=' * 40)

    #  ============== Part 4: Predict and Accuracies ==============
    sample = np.array([1, 45, 85])
    prob = sigmoid(np.dot(sample, theta))
    print('For a student with scores 45 and 85, we predict an admission\
    	       probability of ', prob)
    print('Expected value: 0.775 +/- 0.002\n\n')
    # compute accuracy on our training set
    p = predict(theta, X)
    accuracy = np.mean(np.double(p == y)) * 100
    print('Train Accuracy:', accuracy)
    print('Expected accuracy (approx): 89.0')
    print('=' * 40)



    #  ============== Part 5: 第二部分 非线性数据 ==============
    ex2_file2 = './ex2data2.txt'
    data = np.loadtxt(ex2_file2, delimiter=',')

    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    y = np.reshape(y, (y.shape[0], 1))  # 转换为列向量

    plotData(X, y, True)

    import pandas as pd
    data2 = pd.read_csv(ex2_file2, header=None, names=['Test1', 'Test2', 'Accepted'])

    # Feature mapping 特征映射：做特征工程
    degree = 5  # 总共n = 11维：
    x1 = data2['Test1']
    x2 = data2['Test2']

    data2.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(i):
            print('x1^', i - j, '; x2^', j)
            # x1^ 1 ; x2^ 0
            # x1^ 2 ; x2^ 0
            # x1^ 1 ; x2^ 1
            # x1^ 3 ; x2^ 0
            # x1^ 2 ; x2^ 1
            # x1^ 1 ; x2^ 2
            # x1^ 4 ; x2^ 0
            # x1^ 3 ; x2^ 1
            # x1^ 2 ; x2^ 2
            # x1^ 1 ; x2^ 3
            data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

    data2.drop('Test1', axis=1, inplace=True)
    data2.drop('Test2', axis=1, inplace=True)

    cols = data2.shape[1]
    X2 = data2.iloc[:, 1:cols]
    y2 = data2.iloc[:, 0:1]

    X = np.array(X2.values)
    y = np.array(y2.values)
    print(np.shape(X))  # (118, 11)
    print(np.shape(y))  # (118, 1)

    (m, n) = X.shape
    initial_theta = np.zeros((n, 1))
    iterations = 400000
    theta = gradDescent(initial_theta, X, y, 0.01, iterations)

    p = predict(theta, X)
    accuracy = np.mean(np.double(p == y)) * 100
    print('Train Accuracy:', accuracy)  # 73.72881355932203
    print('Expected accuracy (approx): 89.0')
    print('=' * 40)


    X2 = np.array(X2.values)
    y2 = np.array(y2.values)
    learningRate = 1
    theta2 = np.zeros(11)
    result2 = op.fmin_tnc(func=costFunction, x0=theta2, fprime=gradient, args=(X2, y2, learningRate))

    theta = np.matrix(result2[0])
    probability = sigmoid(X2 * theta.T)
    predictions = [1 if x >= 0.5 else 0 for x in probability]
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))  # accuracy = 91%