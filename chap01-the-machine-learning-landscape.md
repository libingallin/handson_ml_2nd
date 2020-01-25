# Main contents

1.  [What is machine learning?](#1-what-is-machine-learning)
2.  [Why user ML?](#2-why-use-ml)
3.  [Types of ML systems](#3-types-of-ml-systems)
    -   [Supervised/Unsupervised learning](#31-supervisedunsupervised-learning)
        -   [Supervised learning](#311-supervised-learning)
        -   [Unsupervised learning](#312-unsupervised-learning)
        -   [Semisupervised learning](#313-semi-supervised-learning)
        -   [Reinforcement learning](#314-reinforcement-learning)
    -   [Batch and online learning](#32-batch-and-online-learning)
        -   [Batch learning](#321-batch-learning)
        -   [Online learning](#322-online-learning)
    -   [Instance-based v.s. model-based learning](#33-instance-based-vs-model-based-learning)
        -   [Instance-based learning](#331-instance-based-learning)
        -   [Model-based learning](#332-model-based-learning)

4.  [Main challenges of ML](#4-main-challenges-of-ml)
    -   [Bad data](#41-bad-data)
        -   [Insufficient quantity of training data](#411-insufficient-quantity-of-training-data)
        -   [Nonrepresentative training data](#412-nonrepresentative-training-data)
        -   [Poor quality data](#413-poor-quality-data)
        -   [Irrelevant features](#414-irrelevant-features)
    -   [Bad algorithms](#42-bad-algorithms)
        -   [Overfitting](#421-overfitting-the-training-data)
        -   [Underfitting](#422-underfitting-the-training-data)

# 1. What is Machine Learning?

ML is the science (and art) of programming computers so they can learn from data.

+   **a slightly more generally definition**:

    ML is the filed of study that gives computers the ability to learn without being explicitly programmed.

+   **a more engineering-oriented** (以...为导向的/目标的) **one**:

    A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

    [Back to top :arrow_up:](#main-contents)

# 2. Why use ML?

+   Problems for which existing solutions require a lot of fine-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better than the traditional approach.

+   Complex problems for which using a traditional approach yields no good solution: the best Machine Learning techniques can perhaps find a solution.

+   Fluctuating environments: a Machine Learning system can adapt to new data.

+   Getting insights about complex problems and large amounts of data.



![The traditioanl approach](./figs/chap01-figs/traditional-program.png)
<p align="center">The traditional approach</p>

![The ML approach](./figs/chap01-figs/ml-program.png)
<p align="center">The ML approach</p>

![ml-automatically-adapting-to-change](./figs/chap01-figs/ml-automatically-adapting-to-change.png)
<p align="center">Automatically adapting to change</p>

![ml-can-help-humans-learn](./figs/chap01-figs/ml-can-help-humans-learn.png)
<p align="center">ML can help humans learn</p>

[Back to top :arrow_up:](#main-contents)

# 3. Types of ML systems

-   Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning).
-   Whether or not they can learn incrementally on the fly (online versus batch learning)
-   Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)

## 3.1 Supervised/Unsupervised Learning

Machine Learning systems can be classified according to the amount and type of supervision they get during training.

### 3.1.1 Supervised Learning

The training set you feed to the algorithm includes the desired solutions, called *labels*.

![supervised-learning-example](./figs/chap01-figs/supervised-learning-example.png)
<p align="center">Supervised Learning (classification)</p>

- classification
- regression

**Note that some regression algorithms can be used for classification as well, and vice versa** (e.g.,  *Logistic Regression* is commonly used for classification, outputing a value that corresponds to the probability of belonging to a given class.

### 3.1.2 Unsupervised Learning

The training set is unlabeled.
    
![dataset-for-unsupervised-learning](./figs/chap01-figs/dataset-for-unsupervised-learning.png)
<p align="center">Dataset for Unsupervised Learning</p>

-   Clustering: detect groups
    -   K-Means
    -   DBSCAN
    -   Hierarchical Cluster Analysis (HCA): may also subdivide each group into smaller groups
-   Anomaly detection and novelty detection
    -   One-class SVM
    -   Isolation Forest
-   Visualization and dimensionality reduction
    -   Principal Component Analysis (PCA)
    -   Kernel PCA
    -   Locally Linear Embedding (LLE)
    -   t-Distributed Stochastic Neighbor Embedding (t-SNE)
-   Association rule learning
    -   Apriori
    -   Eclat
-   **Notes**
    -   *Visualization algorithms*: output a 2D or 3D representation of your data that can easily be plotted. These algorithms try to preserve as much structure as they can (e.g., trying to keep separate clusters in the input space from overlapping in the visualization) so that you can understand how the data is organized and perhaps identify unsuspected patterns.
    -   *dimensionality reduction*: simplify the data without losing too much information. One way to do this is to merge several correlated features into one, called *feature exraction*.
    -   *anomaly detection*: The system is shown mostly normal instances during training, so it learns to recognize them; then, when it sees a new instance, it can tell whether it looks like a normal one or whether it is likely an anomaly.
    -   *novelty detection*: aims to detect new instances that look different from all instances in the training set. This requires having a very “clean” training set, devoid of (缺乏) any instance that you would like the algorithm to detect.
    -   *association rule learning*: dig into large amounts of data and discover interesting relations between attributes.

### 3.1.3 Semi-supervised Learning

There has plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s partially labeled. This is called *semisupervised learning*.

![semi-supervised-learning](./figs/chap01-figs/semi-supervised-learning.png)
<p align="center">Semisupervised learning with two classes (triangles and squares): the unlabeled examples (circles) help classify a new instance (the cross) into the triangle class rather than the square class, even though it is closer to the labeled squares</p>

**Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms**. For example, *deep belief networks* (DBNs) are based on unsupervised components called *restricted Boltzmann machines* (RBMs) stacked on top of one another. RBMs are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques.

### 3.1.4 Reinforcement Learning

The learning system, called an *agent* in this context, can observe the environment, select and perform actions, and get *rewards* in return (or *penalties*). It must then learn by itself what is the best strategy, called a *policy*, to get the most reward over time.

A policy defines what action the agent should choose when it is in a given situation.

![reinforcement-learning](./figs/chap01-figs/reinforcement-learning.png)
<p align="center">Reinforcement Learning</p>

[Back to top :arrow_up:](#main-contents)

## 3.2 Batch and Online Learning

Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

### 3.2.1 Batch Learning

In *batch learning*, the system is **incapable of learning incrementally**: it must be trained **using all the available data** (generally takes a lot of time and computing resources, and is typically done offline).  After the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called **offline learning**.

+   If you **want a batch learning system to know about new data** (such as a new type of spam), you **need to train a new version of the system from scratch on the full dataset** (not just the new data, but also the old data), then stop the old system and replace it with the new one.

    This solution is simple and often works fine, but **training using the full set of data can take many hours**, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data (e.g., to predict stock prices), then you need a more reactive solution.

+   Also, **training on the full set of data requires a lot of computing resources** (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm.

+   Finally, if your system needs to be able to learn autonomously (独立地) and it has limited resources (e.g., a smartphone application or a rover on Mars), then carrying around large amounts of training data and taking up a lot of resources to train for hours every day is a showstopper.

A better option in all these cases is to use algorithms that are capable of learning incrementally.

### 3.2.2 Online Learning

In *online learning*, you train the system **incrementally by feeding it data instances sequentially, either individually or in small groups called *mini-batches***. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

![online learning](./figs/chap01-figs/online-learning.png)
<p align="center">In online learning, a model is trained and launched into production, and then it keeps learning as new data comes in</p>

+   Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously (独立地). 
+    It is also a good option **if you have limited computing resources**: once an online learning system has learned about new data instances, it does not need them anymore, so you can discard them (unless you want to be able to roll back to a previous state and “replay” the data). This can save a huge amount of space.
+   Online learning algorithms can also be used to **train systems on huge datasets** that cannot fit in one machine’s main memory (this is called **out-of-core learning**). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

>   **Out-of-core learning** is usually done offline (i.e., not on the live system), so online learning can be a confusing name. Think of it as incremental learning.

![use-online-learning-to-handle-huge-datasets](./figs/chap01-figs/use-online-learning-to-handle-huge-datasets.png)
<p align="center">Using online learning to handle huge datasets</p>

**One important parameter of online learning systems** is how fast they should adapt to changing data: this is called the **learning rate**. If you *set a high learning rate*, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data (you don’t want a spam filter to flag only the latest kinds of spam it was shown). Conversely, if you *set a low learning rate*, the system will have more inertia (惰性、惯性); that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points (outliers).

**A big challenge with online learning is that if bad data is fed to the system, the system’s performance will gradually decline**. To reduce this risk, you

+   need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. 
+   may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).

[Back to top :arrow_up:](#main-contents)

## 3.3 Instance-based v.s. Model-based Learning

One more way to categorize Machine Learning systems is by how they *generalize*.

### 3.3.1 Instance-based Learning

The system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them).

![instance-based-learning](./figs/chap01-figs/instance-based-learning.png)
<p align="center">Instance-based learning</p>

### 3.3.2 Model-based Learning

Another way to generalize from a set of examples is to build a model of these examples and then use that model to make *predictions*. This is called *model-based learning*.

![model-based-learning](./figs/chap01-figs/model-based-learning.png)
<p align="center">Model-based learning</p>

[Back to top :arrow_up:](#main-contents)

# 4. Main challenges of ML

Since the main task is to select a learning algorithm and train it on some data, the two things that can go wrong are "bad algorithm" and "bad data".

## 4.1 Bad data

### 4.1.1 Insufficient quantity of training data

**It takes a lot of data for most ML algorithms to work properly**. Even for very simple problems you typically need thousands of examples, and for complex problems such as image or speech recognition you may need millions of examples (unless you can reuse parts of an existing model).

[Microsoft 2001 paper](https://homl.info/6) showed that very different ML algorithms, including fairly simple ones, performed almost identically well on a complex problem of natural language disambiguation once they were given enough data.

![the-importance-of-data-vs-algorithm](./figs/chap01-figs/the-importance-of-data-vs-algorithm.png)
<p align="center">The importance of data v.s. algorithm</p>

These results suggest that we may want to reconsider the trade-off between spending time and money on algorithm development v.s. spending it on corpus development.

[2009 paper](https://homl.info/7) showed that, however, that **small- and medium-sized datasets are still very common, and it is not always easy or cheap to get extra training data⁠—so don’t abandon algorithms just yet**.

### 4.1.2 Nonrepresentative training data

In order to generalize well, **it is crucial that your training data be representative of the new cases you want to generalize to**. This is true whether you use instance-based learning or model-based learning.

It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have **sampling noise** (i.e., nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called **sampling bias**.

### 4.1.3 Poor-quality data

If your training data is full of errors, outliers, and noise (e.g., due to poor-quality measurements), it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well. It is often well worth the effort to spend time cleaning up your training data.

+   Simply discard clearly outliers or try to fix the errors manually.
+   If some instances are missing a few features, you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it.

### 4.1.4 Irrelevant features

Garbage in, garbage out.

ML system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones. A critical part of the success of a ML project is coming up with a good set of features to train on. This process, called **feature engineering**, involves the following steps:

+   *feature selection*: select the most useful features to train on among existing features
+   *feature extraction:* combine existing features to produce a more useful one
+   creating new features by gathering new data

[Back to top :arrow_up:](#main-contents)

## 4.2 Bad algorithms

### 4.2.1 Overfitting the training data

**Overfitting** means that the model performs well on the training data, but it does not generalize well.

**Overfitting happens when the model is too complex relative to the amount and noisiness of the training data**. Here are possible solutions:

+   Simplify the model by selecting one with fewer parameters, by reducing the number of attributes in the training data, or by constraining the model.
+   Gather more training data.
+   Reduce the noise in the training data (e.g., fix data errors and remove outliers).

Constraining a model to make it simpler and reduce the risk of overfitting is called **regularization**. The amount of regularization to apply during learning can be controlled by a *hyperparameter*. If u set the regularization hyperparameter to a very large value,  you will get an almost flat model (a slope close to zero), the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution.

A **hyperparameter** is a parameter of a learning algorithm (not of the model).

-   it is not affected by the learning algorithm itself.
-   it must be set prior to training and remains constant during training.

### 4.2.2 Underfitting the training data

**Underfitting** is the opposite of overfitting: it occurs when your model is too simple to learn the underlying structure of the data.

+   select a more powerful model, with more parameters
+   feed better features to the learning algorithm (feature engineering)
+   reduce the constraints on the model (e.g., reduce the regularization hyperparameter)

[Back to top :arrow_up:](#main-contents)

# 5. Testing and validating

**It is common to use 80% of the data for training and hold out 20% for testing**. However, this depends on the size of the dataset: if it contains 10 million instances, then holding out 1% means your test set will contain 100,000 instances, probably more than enough to get a good estimate of the generalization error.

**validation set == development set == dev set**

## 5.1 Data mismatch

In some cases, it’s easy to get a large amount of data for training, but this data probably won’t be perfectly representative of the data that will be used in production.

After training your model, if you observe that the performance of the model on the validation set is disappointing, you will not know whether this is because your model has overfit the training set, or whether this is just due to the mismatch between the training data and the validation data.

One solution is to hold out some of the training data in yet another set that Andrew Ng calls the **train-dev set**. After the model is trained (on the training set, *not* on the train-dev set), you can evaluate it on the train-dev set.

-   If it performs well, then the model is not overfitting the training set. If it performs poorly on the validation set, the problem must be coming from the data mismatch. You can try to tackle this problem by preprocessing training data to make them look more like the practical data, and then retraining the model.
-   If the model performs poorly on the train-dev set, then it must have overfit the training set, so you should try to simplify or regularize the model, get more training data, and clean up the training data.

## 5.2 No Free Lunch (NFL) Theorem

If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.

There is no model that is *a priori* guaranteed to work better (hence the name of the theorem). The only way to know for sure which model is best is to evaluate them all. Since this is not possible, in practice you make some reasonable assumptions about the data and evaluate only a few reasonable models.

[Back to top :arrow_up:](#main-contents)
