# Train Deep NNs

Here are some problems when training a deep DNN:

+   **Vanishing gradients or exploding gradients problem**. This is when the gradients grow smaller and smaller, or larger and larger, when flowing backward through the DNN during training. Both of these problems make lower layers very hard to train.

[Here]() are some most popular solutions such problems.

+   **Not enough training data** for such a large NN, or it might be too costly to label.

[Transfer learning and unsupervised pretraining]() can help tackle complex tasks even with little labeled data.

+   **Training may be extremely slow**.

[Various optimizers]() can speed up training large model tremendously.

+   A model with millions of parameters would severely risk **overfitting** the training set, especially if there are not enough training instances or if they are too noisy.

A few popular [regularization techniques]() are introduces for large NNs.

