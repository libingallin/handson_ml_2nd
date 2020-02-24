# chap09. Unsupervised Learning Techniques

The vast majority of the available data is unlabeled.

>   If intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. (from Yann LeCun)

There is a huge potential in unsupervised learning that we have only barely (仅仅) started to sink your teeth into.

>   sink your teeth into sth 全神贯注做…，集中精力搞…；决心解决…，认真对待…

Here, a few more unsupervised learning tasks and algorithms will be introduced:

1.  **Clustering**

    The goal is to group similar instances together into *clusters*.

    It's a great tool for

    -   data analysis
    -   customer segmentation
    -   recommender systems
    -   search enignes
    -   image segmentation
    -   semi-supervised learning
    -   dimensionality reduction
    -   ...

2.  **Anomaly detection**

    The goal to learn what "normal" data looks like, and then use that to detect abnormal instances.


3.  **Density estimation**

    This is the task of estimating the **probability density function (PDF)** of the random process that generated the dataset.

    It is commonly used for anomaly detection: instances located in very low-density regions are likely to be anomalies.

    It's also useful for data analysis and visualization.

4.  **Codes:**
    -   [start clustering by using K-Means and DBSCAN]()
    -   [discuss Gaussian mixture models and use it for density estimation, clustering, and anomaly detection]()
    -   [exercise]()

