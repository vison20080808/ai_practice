《统计学习方法》李航

支持向量机（support vector machines, SVM）是 一种二分类模型。
基本模型是 定义在特征空间上的间隔最大的线性分类器。间隔最大，使之有别于感知机。
还包括"核技巧"，使之成为 实质上的非线性分类器。
学习策略：间隔最大化。

模型从简到繁：线性可分支持向量机、线性支持向量机、非线性支持向量机。
线性可分支持向量机：硬间隔最大化
线性支持向量机：训练数据近似可分，软间隔最大化来学习
非线性支持向量机：训练数据线性不可分，使用核技巧（kernel trick）及软间隔最大化

核方法（kernel method）是比支持向量机更为一般的机器学习方法。

函数间隔：可以表示分类预测的正确性及确信度。
几何间隔：对分离超平面的法向量w加约束（规范化，比如||w|| = 1）

支持向量机学习的基本想法：求解能准确划分训练数据集并且几何间隔最大的分离超平面。

凸二次规划问题：目标函数 min 0.5 * ||w||^2  ；不等式约束 s.t.  yi(w x + b) - 1 ≥ 0
求解输出：分离超平面 w^T x + b = 0；分类决策函数 f(x) = sign(w^T x + b)

间隔margin：依赖于分离超平面的法向量w，等于 2 / ||w||
支持向量机，由很少的"重要的"训练样本确定（因为支持向量的个数一般很少）

w的维度 == 数据维度


学习的对偶算法（dual algorithm：
优点：更易求解；自然引入核函数
首先构建拉格朗日函数，引进拉格朗日乘子（Lagrange multiplier）alpha i ≥0.
根据拉格朗日对偶性，原始问题的对偶问题变为极大极小问题：对w,b求min；对alpha 求max
求解alpha，从而算得：w = alpha y x；  b = y - alpha y (xi xj)

软间隔最大化：
对每个样本点引入一个松弛变量（slack variable）；惩罚参数C；
目标函数变为：min 0.5 * ||w||^2 + C * slack之和

线性支持向量机学习:
解释1：学习策略为软间隔最大化，学习算法为凸二次规划；
还有另一种解释2，最小化：正则化的合页损失函数（hinge loss function）

核技巧（kernel trick）基本思想：通过非线性变换将输入空间（欧式空间or离散集合）对应于一个特征空间（希尔伯特空间H）；使得输入空间的超曲面 -> 特征空间中的超平面。
在学习与预测中，只定义核函数K(x, z)，而不显式定义映射函数φ(x)。 K(x, z) = φ(x) * φ(z)
特征空间、映射函数的 取法并不唯一。

对偶问题中，目标函数和决策函数都有，实例之间的内积，可以用核函数替代。等价于将输入空间中的内积x1x2，变换为特征空间中的内积φ(x1)φ(x2)
通常说的核函数，就是正定核函数（positive definite kernel function）
实际应用中，往往应用已有的核函数，比如：
1、多项式核函数（polynomial）：对应的SVM是p次多项式分类器;
2、高斯核函数（Gaussian）：对应高斯径向基函数（radial basis function）分类器
3、字符串核函数（string）：离散数据集合


序列最小最优化算法（sequential minimal optimization, SMO），1998年Platt提出。
解凸二次规划的对偶问题，变量是拉格朗日乘子αi，变量总数等于训练 样本容量N。
基本思路：如果所有变量的解都满足KKT条件，那么问题就解了。否则，选择两个变量，固定其它变量。针对这两个变量构建一个二次规划问题。

子问题的两个变量，只有一个是自由变量。因为，一个确定了，另一个也随之确定。
变量的选择方法：第1个变量（外层循环）选取违反KKT条件最严重的样本点。第2个变量（内层循环）的标准是希望能使α2有足够大的变化（由约束条件自动确定）。
每次完成两个变量的优化后，都要重新计算阈值b

总结：
一种SVM学习的快速算法；
采用启发式的方法：不断将原二次规划问题分解为只有两个变量的二次规划子问题，并对子问题进行解析求解，直到所有变量满足KKT条件为止。




《机器学习实战 Machine Learning in Action》

有些人认为，  SVM是最好的现成的分类器。现成的：分类器不加修改即可直接使用。

SVM有很多实现，最流行的一种：序列最小优化（Sequential Minimal Optimization, SMO）算法。

分隔超平面（separating hyperplane）: N-1维（其中，N为数据集的维度）；可以写成w^T x + b

支持向量（support vector）就是离超平面最近的那些点。


"机"：因为它会产生一个二值决策结果，是一种决策"机"。

核函数的工具，将数据转换成易于分类器理解的形式。

最流行的核函数：径向基函数(radial basis function)。

把核函数理解成 包装器or接口。

SVM 中所有的运算都可以写成内积（inner product, 也称为点积）的形式。

将内积替换成核函数的方式，称为 核技巧（kernel trick）或 核"变电"（kernel substation）.

应用简化版SMO算法：几百个点的小规模数据上。

完整版 Platt SMO算法：选择alpha的方式不同，提速的启发方法。

基于SVM的手写数字识别，径向基核函数。

