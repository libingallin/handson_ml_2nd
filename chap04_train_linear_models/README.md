# <p align="center"><i>Train Modles</i></p>

虽然很多时候，不需要知道模型的实现细节（implementation details），但理解模型是如何工作的（understand what's under the hood）有很多帮助，针对你的任务：

- 快速找到（quickly home in）合适的模型

- 正确的训练算法

- 恰当的超参数

- 高效地 debug 问题和分析错误

<br>

**这里面的讨论的大部分 topics 对理解（understand）、构建（build）和训练（train）神经网络（neural networks）是至关重要的（essential）。**

<br>

首先，从最简单的 **线性模型（Linear Regression, LR）** 开始，并讨论 2 种不同的方法来训练线性模型：

- 通过 **闭式方程（closed-form equation）** 直接计算出最适合训练集的模型参数——使在训练集上损失函数（cost function）最小的模型参数。

- 使用称之为 **梯度下降（Gradient Descent, GD）** 的迭代优化方法来逐渐调整模型参数直至在训练集上的损失函数（cost function）最小，并最终收敛到（converge to）第一种方法得到的参数。而且，会介绍几种梯度下降的变体（variants）—— **Batch GD**、**Mini-Batch GD** 和 **Stochastic GD**，这些在深度学习中会频繁用到。

接下来，会介绍一个比较复杂的模型——**多项式回归（Polynomial Regression）**，适用于非线性数据集（nonlinear datasets）。由于这个模型的参数比 LR 更多，因此更容易（more prone）过拟合训练集——使用 **学习曲线（Learning curves）** 来判断是否发生。随着，会介绍几种不同的正则化技巧（regularization techniques），来减少过拟合的风险。

最后，学习两种常用于分类任务的模型：

- **Logistic Regression**

- **Softmax Regression**
