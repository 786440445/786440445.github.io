---
title: tensorflow之实现神经网络常见层
catalog: true
toc_nav_num: true
date: 2019-05-09 15:50:42
subtitle: 卷积，池化，激活，全链接
header-img:
tags: tensorflow
---

# 卷积原理
1. 输入矩阵格式：四个维度，依次为：样本数、图像高度、图像宽度、图像通道数
1. 输出矩阵格式：与输出矩阵的维度顺序和含义相同，但是后三个维度（图像高度、图像宽度、图像通道数）的尺寸发生变化。
1. 权重矩阵（卷积核）格式：同样是四个维度，但维度的含义与上面两者都不同，为：卷积核高度、卷积核宽度、输入通道数、输出通道数（卷积核个数）
1. 输入矩阵、权重矩阵、输出矩阵这三者之间的相互决定关系
1. 卷积核的输入通道数（in depth）由输入矩阵的通道数所决定。（红色标注）
1. 输出矩阵的通道数（out depth）由卷积核的输出通道数所决定。（绿色标注）
1. 输出矩阵的高度和宽度（height, width）这两个维度的尺寸由输入矩阵、卷积核、扫描方式所共同决定。计算公式如下。（蓝色标注）
![](/img/article/filter_func.png)

# 代码分析
1. tensorflow层是四维设计的,[batch_size, width, height, channels]，input4d.shape=(1, 1, 25, 1)，本例中，批量大小为1，宽度为1，高度为25，颜色通道为1
1. 扩展维度expand_dims(),降维squeeze()，
1. 卷积层结果维度公式：output_size = (W - F + 2P)/S + 1
1. W是输入数据维度，F是过滤层大小，P是padding大小，S是步长
1. filter的维度=(1, 5, 1, 1)，过滤器大小为1x5，输入通道为1， 输出通道（即卷积核个数）为1

## 卷积层-1
输入[1, 1, 25, 1]，w=[1, 5, 1, 1]，输出为[1, 1, 21, 1]

## 池化层-1
1. 池化层和卷积层类似，但是没有过滤层，只有形状，步长，和padding选项
1. 输入[1, 1, 21, 1]，池化过滤器大小[1, 1, 5, 1]，输出为[1, 1, 17, 1]

## 全链接层
1. 全链接weight_shape=[17, 5]，所以输入为[1, 1, 17, 1]，通过 tf.squeeze压缩到[1, 17]，最后得到shape=[1, 5]

## expand_dims用法
![](/img/article/expand_dims_test.png)
1. tf.expand_dims(Matrix, axis) 即在第axis维度处添加一个维度
1. 如上图, input1.shape = (5), tf.expand_dims(input1d, 0)即在第0个维度加一个即shape=(1,5)
1. 在input3,shape=(1,1,5)情况下，调用tf.expand_dims(input3d, 3)即在第三个维度处添加一个即为shape=(1, 1, 5, 1)

## squeeze用法
squeeze(
    input,
    axis=None,
    name=None,
    squeeze_dims=None
)
类似expand_dims,他是删除维度为1的所有维度，或者指定维度(维度必须为1才能删除)



```python
import tensorflow as tf
import numpy as np
sess = tf.Session()
# 初始化数据，长度为25
data_size = 25
data_1d = np.random.normal(size=data_size)
print('Input data: ')
print(data_1d)
x_input_id = tf.placeholder(dtype=tf.float32, shape=[data_size])

# 定义一个卷积层函数，声明一个随机过滤层
def conv_layer_1d(input_1d, my_filter):
    # 将输入扩展维度为4维，【batch_size, width, height, channels】
    # 输出维度为output_size = (W - F + 2P)/S + 1
    # W:输入数据维度
    # F:过滤层大小
    # P:padding大小
    # S:步长大小
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    convolution_output = tf.nn.conv2d(input_4d,
                                      filter=my_filter,
                                      strides=[1, 1, 1, 1],
                                      padding='VALID')
    conv_output_1d = tf.squeeze(convolution_output)
    return(conv_output_1d)

# 随机生成一个过滤层窗口大小
my_filter = tf.Variable(tf.random_normal(shape=[1, 5, 1, 1]))
# 卷积层输出结果
my_convolution_output = conv_layer_1d(x_input_id, my_filter)

# 声明一个激活函数
def activation(input_1d):
    return(tf.nn.relu(input_1d))

# 卷积层输出后经过激活函数的结果
my_activation_output = activation(my_convolution_output)

# 声明池化层函数
def max_pool(input_1d, width):
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    pool_out = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')
    pool_output_1d = tf.squeeze(pool_out)
    return(pool_output_1d)

my_maxpool_output = max_pool(my_activation_output, width=5)

# 最后一层连接的是全链接层
def fully_connected(input_layer, num_outputs):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])

    input_layer_2d = tf.expand_dims(input_layer, 0)
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    full_output_1d = tf.squeeze(full_output)
    return full_output_1d

my_full_output = fully_connected(my_maxpool_output, 5)

# 初始化所有变量,运行计算图打印每层输出结果
init = tf.global_variables_initializer()
sess.run(init)
feed_dict = {x_input_id: data_1d}

# 卷积层输出
print('Input = array of length 25')
print('Convolution w/filter, length = 5, stride_size = 1, result in an array of length 21: ')
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# 激活函数输出
print('Input = the above array of length 21')
print('ReLU element wise returns the array of length 21: ')
print(sess.run(my_activation_output, feed_dict=feed_dict))

# 池化层输出
print('Input = the above array of length 21')
print('MaxPool, window length = 5, stride size = 1, results in array of length 17: ')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# 全链接层输出
print('Input = the above array of length 17')
print('Fully connected layer on all four rows with five outputs: ')
print(sess.run(my_full_output, feed_dict=feed_dict))
```


