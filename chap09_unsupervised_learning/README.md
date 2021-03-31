虽然目前大多数的 ML 应用都基于监督学习（supervised learning）——因此是大部分的投资方向，但是绝大多数的数据都是未标记的（unlabeled）——有输入特征X，但没有标签y。Yann LeCun 说过

> If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake.

也就是说，无监督学习这块有着巨大的潜力（potential），而我们才刚刚接触到（sink our teeth into）。

<br>

**聚类（Clustering）** 的目标是将相似的样本分组到 clusters 中。聚类是一个很好的工具，适用于数据分析（data analysis）、客户分群（customer segmentation）、推荐系统（recommender system）、搜索引擎（search engine）、图像分割（image segmentation）、半监督学习（semi-supervised learning）、降维（dimensionality reduction）等。

<br>

**异常检测（Anomaly detection）** 的目的学习”normal“的数据看起来是怎样的，然后将其用于检测异常样本（abnormal instances），如生产线上的缺陷产品（defective items on a production line）或时间序列中（time series）的新趋势（new trend）。

<br>

**密度估计（Density estimation）**——这是用于估计生成数据集的随机过程（random process）的概率密度函数（Probability Density Function, PDF）的任务。**通常与用于异常检测**——位于低密度区域的样本很可能是异常值（anomalies）。对于数据分析和数据可视化也很有用。

> This is the task of estimating the probability density function (PDF) of the random process that generated the dataset.

<br>

-   [聚类——KMeans 和 DBSCAN]()

-   [了解如何将高斯混合模型用于密度估计、聚类和异常检测](