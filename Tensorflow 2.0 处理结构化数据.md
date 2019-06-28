# Tensorflow 2.0 处理结构化数据

- ### 简介

  tf 2.0是tf的最新版本，这次将演示如何对结构化数据如csv表格进行分类

  - 使用pandas加载CSV文件
  - 构建一个输入管道，使用tf.data批处理和shuffle（打乱数据顺序）
  - 从CSV中的列columns映射到特征features以进行模型训练
  - 用Keras来创建、训练和评价模型

- ### 数据集

  本次使用的数据集是克列弗兰诊所提供的只有击败啊很难过的数据，每一行代表一个患者，每列代表患者的一个属性，我们将使用这些属性来预测患者是否患有心脏病，显而易见，本次的分类任务是一个二分类任务

- ### 数据分析

  - 导入库tensorflow和其他的库

    注意在colab中`!`后面的代码表示是在命令行中运行的，在这里是安装tf 2.0的命令，如果已经安装了会显示requirement is already satisfied.

    ~~~python
    from __future__ import absolute_import, division, print_function, unicode_literals
    
    import numpy as np
    import pandas as pd
    
    !pip install -q tensorflow==2.0.0-beta1
    import tensorflow as tf
    
    from tensorflow import feature_column
    from tensorflow.keras import layers
    ~~~

  - 使用pandas来创建DataFrame数据类型，用于稍后的数据格式转换

    ~~~python
    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    dataframe = pd.read_csv(URL)
    # 查看数据的前五行
    dataframe.head()
    ~~~

    |      |  age |  sex |   cp | trestbps | chol |  fbs | restecg | thalach | exang | oldpeak | slope |   ca |       thal | target |
    | ---: | ---: | ---: | ---: | -------: | ---: | ---: | ------: | ------: | ----: | ------: | ----: | ---: | ---------: | -----: |
    |    0 |   63 |    1 |    1 |      145 |  233 |    1 |       2 |     150 |     0 |     2.3 |     3 |    0 |      fixed |      0 |
    |    1 |   67 |    1 |    4 |      160 |  286 |    0 |       2 |     108 |     1 |     1.5 |     2 |    3 |     normal |      1 |
    |    2 |   67 |    1 |    4 |      120 |  229 |    0 |       2 |     129 |     1 |     2.6 |     2 |    2 | reversible |      0 |
    |    3 |   37 |    1 |    3 |      130 |  250 |    0 |       0 |     187 |     0 |     3.5 |     3 |    0 |     normal |      0 |
    |    4 |   41 |    0 |    2 |      130 |  204 |    0 |       2 |     172 |     0 |     1.4 |     1 |    0 |     normal |      0 |

    

  ## 特征工程

  - 切分数据为训练集、验证集和测试集

    ~~~python
    from sklearn.model_selection import train_test_split
    
    # 先切分用于训练的集合和测试集
    train, test = train_test_split(dataframe, test_size=0.2)
    # 再把训练集切分成真正的训练集和验证集
    train, val = train_test_split(train, test_size=0.2)
    # 看看训练集、验证集和测试集各有多少个样本
    print('train example',len(train))
    print('validation example',len(val))
    print('test example',len(test))
    ~~~

    ~~~shell
    193 train examples
    49 validation examples
    61 test examples
    ~~~

  - 定义函数，把pd.DataFrame数据格式转化为为tf.data的batch格式，以便后续的训练

    ~~~python
    # 定义数据转化方法
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        # 这个函数的基于浅拷贝副本进行的转换，意味着原来的dataframe不会被抹去
        dataframe = dataframe.copy()
        # 把y和x分开来
        labels = dataframe.pop('target')
        # 进行数据格式转化
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            # 对数据进行损及打乱，打乱范围是整个数据集
            ds = ds.shuffle(buffer_size=len(dataframe))
        # 同时把数据集变成批BatchDataset形式方便后续训练
        ds = ds.batch(batch_size)
        return ds
    
    # 用这个方法把前面的pd.DataFrame格式的训练集、测试集、验证集进行转化，变成BatchDataset类型
    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    ~~~

  - 查看经过转换后的数据是怎么样的

    ~~~python
    for feature_batch, label_batch in train_ds.take(1):
        print('Every feature:', list(feature_batch.key()))
        print('A batch of ages:', feature_batch['age'])
        print('A batch of targets', label_batch)
    ~~~

    ~~~shell
    Every feature: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    A batch of ages: tf.Tensor([67 58 42 68 56], shape=(5,), dtype=int32)
    A batch of targets: tf.Tensor([0 1 0 0 0], shape=(5,), dtype=int32)
    ~~~

  - Numeric columns 特征数字化

    光转化数据格式是不够的，模型除了要求数据格式外，对元素的数据格式也是有要求的，必须是数字，但是特征里面的dtype有的是string，要把他们全部转化成数字才可以

    定义转化函数

    ~~~python
    # 从样本中取出一段作为demo使用
    example_batch = next(iter(train_ds))[0]
    
    def demo(feature_column):
        feature_layer = layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())
        
    age = feature_column.numeric_column('age')
    
    demo(age)
    ~~~

    ~~~shell
    [[67.]
     [58.]
     [42.]
     [68.]
     [56.]]
    ~~~

    - 数值化之数据分箱+one-hot分类

      对于类型为连续型数字的属性，类如年龄，不可能每个具体年龄都作为一个属性类别，这样将会使得这个特征具有很多的类别，所以需要对这类特征进行数据分箱，注意下面的one-hot值表示了年龄属于哪个范围

      这里用到的一个方法是bucketized_column()

      ~~~python
      age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
      demo(age_buckets)
      ~~~

      ~~~shell
      [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
       [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
       [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
      ~~~

    - 数值化之one-hot分类

      对于thal特征，先使用categorical_column_with_vocabulary_list()进行数值映射，然后创建新的属性记录one-hot编码

      ~~~python
      thal = feature_column.categorical_column_with_vocabulary_list(
            'thal', ['fixed', 'normal', 'reversible'])
      
      thal_one_hot = feature_column.indicator_column(thal)
      
      # 取样查看数值化情况
      demo(thal_one_hot)
      ~~~

      ~~~python
      [[0. 0. 1.]
       [0. 0. 1.]
       [0. 0. 1.]
       [0. 0. 1.]
       [0. 0. 1.]]
      ~~~

    - 数值化之embedding

      你会发现，thal处理的只是三个选择的特征，但是如果选择太多的话，新增的one-hot列会变得很多，这样one-hot就不是个好的解决方案，此时embedding是个更好的选择

      下面同样用thal这个特征为例子进行embedding嵌入

      ~~~python
      # 指定维度，这样每个无论这个属性有多少个选择，表示这个属性的向量长度都只有8
      thal_embedding = feature_column.embedding_column(thal, dimension=8)
      
      # 取样查看数值化的情况
      demo(thal_embedding)
      ~~~

      ~~~python
      [[0.13279669 0.19413401 -0.69587415 -0.6805197 0.3184564 0.45431668 
        -0.13196784 -0.57410216] 
       [0.13279669 0.19413401 -0.69587415 -0.6805197 0.3184564 0.45431668 
        -0.13196784 -0.57410216] 
       [0.13279669 0.19413401 -0.69587415 -0.6805197 0.3184564 0.45431668 
        -0.13196784 -0.57410216] 
       [0.13279669 0.19413401 -0.69587415 -0.6805197 0.3184564 0.45431668 
        -0.13196784 -0.57410216] 
       [0.13279669 0.19413401 -0.69587415 -0.6805197 0.3184564 0.45431668 
        -0.13196784 -0.57410216]]
      ~~~

    - 数值化之hash分箱

      表示具有大量选项的特征还有一种方法是使用hash，但是有个确定就是容易产生冲突，当然冲突的可能性会比较少，优点是hash_bucket_size会远远小于实际的类别数，以节省空间

      同样以thal特征为例子

      ~~~python
      thal_hashed = feature_column.categorical_column_with_hash_bucket(
            'thal', hash_bucket_size=1000)
      demo(feature_column.indicator_column(thal_hashed))
      ~~~

      ~~~shell
      [[0. 0. 0. ... 0. 0. 0.]
       [0. 0. 0. ... 0. 0. 0.]
       [0. 0. 0. ... 0. 0. 0.]
       [0. 0. 0. ... 0. 0. 0.]
       [0. 0. 0. ... 0. 0. 0.]]
      ~~~

      

  - 特征交叉

    将多个特征组合成单个特征也叫特征交叉，使得模型能偶为每个特征组合学习单独的权重，下面以age和thal特征作为例子，使用crossed_column()方法

    注意两个特征之间的所有可能的组合情况不能都被考虑到，而是把年龄hashed_column后在与thal进行组合，所以你可以指定hash分箱的size大小

    ~~~python
    crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    demo(feature_column.indicator_column(crossed_feature))
    ~~~

    ~~~shell
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    ~~~

  - 特征选择

    选择对结果最有意义的几个特征进行模型训练

    ~~~python
    feature_columns = []
    
    # 选择数值型特征
    for header in ['age','trestbps','chol','thalach','oldpeak','slope','ca']:
        feature_columns.append(feature_column.numeric_column(header))
        
    # 选择数值分箱型特征
    age_buckets = feature_column.bucketized_column(age, boundaries[18, 25, 30, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)
    
    # 选择符号型特征并进行独热编码
    thal = feature_column.categorical_column_with_vocabulary_list(
          'thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)
    
    # 选择符号型特征并进行嵌入编码
    thal_embedding = feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)
    
    # 交叉特征，把数值型特征分箱后进行度热编码
    crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    crossed_feature = feature_column.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)
    
    ~~~

    

  # 构建训练模型

  - 创建特征编码层

    使用keras创建一个全连接层模型

    ~~~python
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    ~~~

  - 调整batch大小进行训练

    ~~~python
    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    ~~~

  - 创建模型，编译模型和训练模型

    ~~~python
    # 创建模型
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)
    
    # 训练模型
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5)
    ~~~

    ~~~shell
    Epoch 1/5
    7/7 [==============================] - 1s 142ms/step - loss: 2.5852 - accuracy: 0.5992 - val_loss: 1.5481 - val_accuracy: 0.6735
    Epoch 2/5
    7/7 [==============================] - 0s 30ms/step - loss: 1.4630 - accuracy: 0.5475 - val_loss: 0.8428 - val_accuracy: 0.6735
    Epoch 3/5
    7/7 [==============================] - 0s 31ms/step - loss: 0.6788 - accuracy: 0.7359 - val_loss: 0.8275 - val_accuracy: 0.6531
    Epoch 4/5
    7/7 [==============================] - 0s 30ms/step - loss: 0.8789 - accuracy: 0.6067 - val_loss: 0.7656 - val_accuracy: 0.6327
    Epoch 5/5
    7/7 [==============================] - 0s 32ms/step - loss: 0.6756 - accuracy: 0.6843 - val_loss: 0.7049 - val_accuracy: 0.6735
    
    <tensorflow.python.keras.callbacks.History at 0x7fbccd338668>
    ~~~

  - 最后用测试集合评估模型

    ~~~python
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy:", accuracy)
    ~~~

    ~~~shell
    2/2 [==============================] - 0s 18ms/step - loss: 0.4494 - accuracy: 0.7869
    Accuracy: 0.78688526
    ~~~

  - 注意

    如果你有更大量的数据集进行深度学习，你将会得到最佳的结果，而使用像这样的小数据集的时候，建议还是使用决策树或者随机森林作为基评估器，本教程的目的不是为了训练一个高准确度的模型，而是演示了使用机构化数据的机制，因此再将来使用自己的数据集时候需要使用代码作为起点。

    