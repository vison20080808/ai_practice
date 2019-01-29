《统计学习方法》李航

EM算法：1977年，一种迭代算法，用于具有隐变量的概率模型参数的 极大似然估计。

每次迭代：
1、E步，求期望expectation；
2、M步，求极大maximization。

EM（expectation maximization，期望极大算法）

概率模型，有时既有观测变量（observable variable），又含有隐变量or潜在变量（latent variable）。


三硬币模型：
观测变量Y：结果0or1；
隐变量Z：第一步掷硬币A的结果（未观测到的）。

Y与Z连在一起，称为 完全数据（complete-data）。

没有解析解，只能通过迭代求解。

Q函数：EM算法核心
完全数据的 log P(Y, Z | θ)，关于在给定Y和当前参数θ(i)下，对未观测数据Z的条件概率分布P（Z | Y, θ(i)）的期望。

EM算法的最大优点：简单性、普适性。

一个重要应用：高斯混合模型(Gaussian misture model)的参数估计
1、明确隐变量，写出完全数据的对数似然函数；
2、E步：确定Q函数；
3、M步：求Q函数对θ的极大值；
4、重复2、3步，直到收敛。

EM算法的推广：
可以解释为：F函数的极大-极大算法。（EM的每次迭代，可由它实现）
基于这个解释：广义期望极大（generalized expection maximization, GEM）算法。

