# Using code examples from keras.io

Code examples documented on keras.io will work fine with `tf.keras`, but you need to change the imports.

For example, consider this **keras.io code**:

```python
from keras.layers import Dense
output_layer = Dense(10)
```

:triangular_flag_on_post:You must change the imports like this (**`tf.keras`**):

```python
from tensorflow.keras.layers import Dense
output_layer = Dense(10)
```

Or, simply use full paths, if you prefer:

```python
# This is more verbose, but can easily see which packages to use,
# and to avoid confusion between standard classes and custom classes.
from tensorflow import keras
output_layer = keras.layers.Dense(10)
```

This is more verbose, but can easily see which packages to use and to avoid confusion, between standard classes and custom classes.

**In production code, I (author of this book) prefer the previous approach**. Many people also use `from tensorflow.keras import layers` followed by `layers.Dense(10)`.
