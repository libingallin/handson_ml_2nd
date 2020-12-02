<font color='red' size=5>Scikit-Learn Design</font>

Scikit-Learn's API is remarkably well designed. The **main design principles** are:

1. **Consistency**: All objects share a consistent and simple interface:
  + <font color='red'>Estimators</font>. Any object that can estimate some parameters based on a dataset is called an *estimator* (e.g., an `imputer` is an estimator). The estimation itself is performed by the `fit()` method, and it takes only a dataset as a parameter (or two for supervised learning algorithms; the second dataset contains the labels). Any other parameter needed to guide the estimation process is considered a <font color='blue'>hyperparameter</font> (such as an *imputer's strategy*), and it must be set as an instance variable (generally via a constructor parameter).
  + <font color='red'>Transformers</font> Some estimators (such as an `imputer`) can also transform a dataset; these are called *transformers*. Once again, the API is quite simple: the transformation is performed by the `transform()` method with the dataset to transform as a parameter. It returns the tranformed dataset. This transformation generally relies on the learned parameters, as is the case for an *imputer*. **All transformers also have a convenience method called `fit_transform()`** that is equivalent to calling `fit()` and then `transform()` (but **sometimes `fit_transform()` is optimized and runs much faster**).
  + <font color='red'>Predictors</font> Finally, some estimators are capable of making predictions given a dataset; they are called *predictors*. A predictor has a `predict()` method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also have a `score()` method that measures the quality of the predictions given a test set.
  
2. **Inspection**
 <font color='blue'>All the esitmator's hyperparameters are accessible directly via public instance variables</font> (e.g., `imputer.strategy`), and <font color='blue'>all the estimator's learned parameters are also accessible via public instance variable with an underscore suffix</font>(e.g., `imputer.statistics_`).

3. **Nonproliferation of classes** Dataset are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hypterparameters are just regular Python strings or numbers.

4. **Composition** Existing building blocks are reused as much as possible. For example, it's easy to create a `Pipeline` estimator from an arbitrary sequence of tranformers followed by a final estimator.

5. **Sensible defaults**. Sklearn provides reasonable default values for most parameters, making it easy to create a baseline working system quickly.