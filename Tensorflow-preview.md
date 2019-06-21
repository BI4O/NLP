# Tensorflow



## Tensorflow安装

###### 需要注意的是，这次安装的是tf2版本，使用的是清华大学的镜像源

- ##### 首先在conda创建同名的虚拟环境

  ~~~shell
  conda create -n tensorflow python=3.7
  ~~~

- 然后激活这个环境

  ~~~shell
  conda activate tensorflow
  ~~~

- 安装tensorflow

  ~~~shell
  pip install tensorflow==2.0.0-alpha0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
  ~~~

- 安装nb_conda

  ~~~shell
  conda install nb_conda
  ~~~

  

## Tensorflow的基本概念

- #### importing required packages

  导入所需要的相关的包

  ~~~python
  import tensorflow as tf
  # tf.enable_edge_execution()
  print(tf.__version__)  # 2.0.0-alpha0
  ~~~

  注意edge execution 可以让使tf的界面交互更丰富，这个在2.0.0版本后已经默认支持

- #### Tensor

  tensor就是张量，就是多维数组，注意 `tf.Tensor` 有两个属性，一个是`shape` 代表这个张量的维度信息，一个是`dtype`也就是数组中的数的类型

  input

  ~~~python
  print(tf.add(1, 2))
  print(tf.add([1, 2], [3, 4]))
  print(tf.square(5))
  print(tf.reduce_sum([1, 2, 3]))
  
  # Operator overloading is also supported
  print(tf.square(2) + tf.square(3))
  ~~~

  output

  ~~~shell
  tf.Tensor(3, shape=(), dtype=int32)
  tf.Tensor([4 6], shape=(2,), dtype=int32)
  tf.Tensor(25, shape=(), dtype=int32)
  tf.Tensor(6, shape=(), dtype=int32)
  tf.Tensor(13, shape=(), dtype=int32)
  ~~~

  

  - ##### Numpy Compatibility

    `tf.Tensor` 与 `np.ndarray`的兼容性非常好，可以不需要经过一个专门的方法转换数据类型就直接参与对方的运算（如乘法），只不过得到的结果会自动转换为运算方法所属的那个包（np/tf）的数据类型

    ###### 首先和回顾一下numpy的乘法

    input

    ~~~python
    ar = np.ones([3,5])
    ar1 = np.ones([5,3])
    # numpy点乘，也就是矩阵乘法
    ar2 = np.dot(ar,ar1)
    # numpy相乘，必须是shape一样，或者直接一个标量
    ar3 = np.multiply(ar,3)
    ar2,ar3
    ~~~

    output

    ~~~python
    (array([[5., 5., 5.],
            [5., 5., 5.],
            [5., 5., 5.]]), 
     array([[3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.]]))
    ~~~

    ###### 看看tensorflow的乘法（注意我还是用的numpy的数据ndarray）

    input

    ~~~python
    ar = np.ones([3,5])  # 当然可以用 tf.ones([3,5]),实测是一样的效果
    ar1 = np.ones([5,3])
    # tf点乘，也就是矩阵的点乘（dot product）
    ar2 = tf.matmul(ar,ar1)
    # tf相乘，必须是元素对元素的相乘，shape必须一样，如果是标量则可以只写一个数
    ar3 = tf.multiply(ar,3)
    ar2,ar3
    ~~~

    output

    ~~~python
    (<tf.Tensor: id=27, shape=(3, 3), dtype=float64, numpy=
     array([[5., 5., 5.],
            [5., 5., 5.],
            [5., 5., 5.]])>, 
     <tf.Tensor: id=30, shape=(3, 5), dtype=float64, numpy=
     array([[3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.]])>)
    ~~~

    

- ### GPU acceleration

  矩阵运算等都可以放在gpu上面

  1. ##### 检查是否可用gpu

     input

     ~~~python
     # 指随机初始化一个3x3的矩阵
     x = tf.random.uniform([3, 3])
     
     print("Is there a GPU available: "),
     print(tf.test.is_gpu_available())
     
     print("Is the Tensor on GPU #0:  "),
     print(x.device.endswith('GPU:0'))
     x
     ~~~

     output

     ~~~python
     Is there a GPU available: 
     False
     Is the Tensor on GPU #0:  
     False
     <tf.Tensor: id=58, shape=(3, 3), dtype=float32, numpy=
     array([[0.6457139 , 0.6058842 , 0.20635498],
            [0.39819157, 0.6746627 , 0.45240736],
            [0.28637683, 0.05444217, 0.01704669]], dtype=float32)>
     ~~~

  2. ##### 指定某些计算在哪个设备上面

     `with tf.device("CPU/GPU:0")`

     格式：

     ~~~python
     # 强制在第一个cpu上执行该task
     print("On CPU:")
     with tf.device("CPU:0"):
       x = tf.random_uniform([1000, 1000])
       assert x.device.endswith("CPU:0")
       time_matmul(x)
     
     # 强制在第一个GPU上执行task,先检查gpu是否可用
     if tf.test.is_gpu_available():
       with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
         x = tf.random_uniform([1000, 1000])
         assert x.device.endswith("GPU:0")
         time_matmul(x)
     ~~~

     