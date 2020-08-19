95% 的问题都可以用 `tf.keras` 和 `tf.data` 解决。但是，lower-level 的 Python API 可用于：

- 在自定义 loss functions、metrics、layers、models、initializers、regularizers、weights constraints 等等时提供 extra control
- 完全控制 training loop，如，对 gradients 应用特殊的 transformations 和 constraints（不仅仅是梯度裁剪），或对网络的不同部分使用不同的 optimizer 等。

<br>

本章还有以下内容：

- [Introduction to TensorFlow](TensorFlow简介.md)
- [Use TensorFlow like NumPy]()
- [Customize models and training algorithms]()
- [TensorFlow functions and graphs]()
- [Exercises]()

