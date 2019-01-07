《统计学习方法》李航
统计学习（statistical learning）也成为统计机器学习（statistical machine learning）

由监督学习（supervised learning）、非监督学习（unsupervised learning）、半监督学习（semi-supervised learning）、强化学习（reinforcement learning）等组成；

三要素：模型（model）、策略（strategy）、算法（algorithm）

回归问题、分类问题、标注问题（输入和输出变量均为变量序列的预测问题）

模型：概率模型或非概率模型。表示形式：条件概率分布 P（Y|X）或 决策函数 Y = f(X)

策略：
损失函数（loss function）或代价函数（cost function）：0-1损失函数、平方、绝对、对数或对数似然损失函数
损失函数的期望：风险函数（risk function）或期望损失（expected loss），是模型关于联合分布P(X, Y)的平均意义下的损失
经验风险（empirical risk）或经验损失（empirical loss）：模型关于训练数据集的平均损失


经验风险最小化（ERM）：可能过拟合（over-fitting）；极大似然估计、多数表决
结构风险最小化（SRM）：等价于正则化（regularization），经验风险+正则化项（regularizer，也叫罚项 penalty term，对模型复杂程度的惩罚）；贝叶斯估计

交叉验证：
样本数据充足，随机划分 训练集（training set）、验证集（validation）、测试集（test）；
样本数据不足：重复使用数据，切分组合为训练集与测试集
简单交叉验证、S折交叉验证（S-fold cross validation）、留一交叉验证（S == N，N为给定数据集的容量）

学习方法的泛化能力（generalization ability）：
由该方法学习到的模型对未知数据的预测能力。
现实中采用最多的办法是通过测试误差来评价
比较泛化误差上界。

监督学习方法 分为 生成方法（generative approach）、判别方法（discriminative）。
所学到的模型 分为 生成模型、判别模型

生成模型：学习P(X, Y)，求出 P(Y | X) 作为预测模型；朴素贝叶斯法、隐马尔科夫模型；因为表示了给定X产生Y的生成关系
判别模型：直接学习f(X) 或 P(Y | X) 作为预测模型；K近邻法、感知机、决策树、逻辑斯蒂回归模型、最大熵模型、支持向量机、提升方法、条件随机场等

分类问题（classification）：
评价分类器性能指标：分类准确率（accuracy）
对于二分类问题：精确率（precision）与召回率（recall）、F1值

标注问题（tagging）:
也是监督学习问题。
方法：隐马尔可夫模型、条件随机场
NLP中的词性标注（part of speech tagging）：给定一个句子，对每个单词进行词性标注，即对一个单词序列预测其对应的词性标记序列

回归问题（regression）：
学习过程等价于函数拟合
最常用的损失函数：平方损失函数。此时，可由最小二乘法求解