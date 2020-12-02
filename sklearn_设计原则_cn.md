<center><b>Scikit-Learn 设计原则</b></center>

Scikit-Learn 的 API 设计得非常好 (remarkably well designed)。下面是主要的设计原则 (design principles)：


- **一致性（Consistency）**

  所有的对象 (object) 保持一致且简单的接口 (interface)。

  - **Estimator**

    可以根据某个数据集估计一些参数的任何对象都称为 **estimator**，如 `SimpleImputer`。估计本身只能由 `fit()` 方法完成，它只能接收数据集 (`X`) 作为参数（如果是监督学习算法，可以有第二个参数——标签(`y`)）。指导/参与估计过程 (guide the estimation process) 的其他任何参数都是超参数 (hyperparameter) （如 `SimpleImputer` 中的 `strategy` 参数），并且必须设置成 instance variable（通常通过一个 constructor parameter）。

  - **Transformer**

    一些 Estimator（如 `SimpleImputer`）也可以用来转换数据，这称之为 **transformers**。API 也很简单：转换 (transformation) 动作由 `transform()` 完成，将需要转换的数据集作为参数。它返回转换后的数据集（ndarray）。转换通常依赖于学到的参数。所有的 transformers 都有一个便捷方方法 `fit_transform()`——相当于先使用 `fit()` 再使用 `transform()`，但有时 `fit_transform()` 经过了优化，运行速度更快。

  - **Predictor**

    给定数据集，一些 estimator 能够做出预测，这种 estimator 称之为 **predictor**。predictor 的 `predict()` 方法接收新数据并返回相应的预测。predictor 的 `score()` 方法在给定测试集的情况下衡量预测的质量 (the quality of the predictions)。

<br>

- **内省（Inspection）**

  estimator 所有的超参数都可以通过公共实例变量 (public instance variables) 来获取，如 `SimpleImputer.strategy`，estimator 所有学到的参数 (learned parameters) 都可以通过带下划线后缀的公共实例变量来获得，如 `SimpleImputer.statistics_`。

<br>

- **Nonproliferation of classes**

  数据集的形式是 NumPy 数组或者 Scipy 的 sparse metrices，而不是自制类 (homemade class)。超参数是普通的 (regular) Python 字符串或者数字。

<br>

- **组成（Composition）**

  尽可能重用现有 building blocks（不要重复造轮子）。如，用以 estimator 结尾的一系列任意的 transformers 来构建一个 pipeline 是很容易的。

<br>

- **合理的默认值（Sensible defaults）**

  Scikit-Learn 为大多数参数提供了合理的默认值 (reasonable default values)，这有助于快速构建一个 baseline 模型。
