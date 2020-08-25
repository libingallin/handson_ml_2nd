到目前为止，使用的 datasets 都可以 fit in memory，但是，很多 DL 系统通常都是在不能 fit in RAM 的大数据集上训练的。

由于 TF 有 **Data API**，TF 可以有效地加载&处理大数据集。只需要创建一个 dataset 对象（dataset object），并告诉它从哪获取数据以及如何 transform 它就可以了。TF 会 take care of 所有的实现细节（implementation details），如 multi-threading、queuing、batching 和 prefetching。而且，**Data API 与 `tf.keras` 无缝配合（work seamlessly）**。

<br>

off the shelf，**Data API 可读取的数据源有：**

-   text files, such CSV files

-   binary files with fixed-size records

-   binary files that use TF's **TFRecord** format, which supports records of varying sizes

>   **TFRecord** is a fexible and efficient binary format usually containing protocol buffers (an open source binary format).

-   SQL databases

-   many open source extensions are available to read from all sorts of data source, such as Google's BigQuery services

<Br>

如何有效地 read 大数据集并不是唯一的麻烦点，通常

-   数据还需要 preprocessing，通常是标准化。

-   数据并不全是方便处理的数值型，可能是 text features、categorical features 等。这些特征需要 encode，如使用 one-hot encoding、bag-of-words 或者 embedding。

解决这些 preprocessing 的方法有：

-   自定义 preprocessing layers

-   使用 Keras 提供的 standard preprocessing layers

<br>

这里包含以下内容：

1.  [Data API](https://nbviewer.jupyter.org/github/libingallin/handson-ml-2nd/blob/master/chap13_load_and_preprocess_data_with_tf/chap13_TF_Data_API.ipynb)

2.  [TFRecord format]()

3.  [如何创建自定义 preprocessing layers 和使用 keras 提供的 preprocessing layers]()

4.  [TF Transform (`tf.Transform`)]()

5.  [TensorFlow Datasets (TFDS)](https://nbviewer.jupyter.org/github/libingallin/handson-ml-2nd/blob/master/chap13_load_and_preprocess_data_with_tf/chap13_tensorflow_datasets.ipynb)
