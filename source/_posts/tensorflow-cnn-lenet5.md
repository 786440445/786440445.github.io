---
title: LeNet5实现MNIST模型
toc_nav_num: true
catalog: true
date: 2019-05-17 16:59:06
subtitle: LeNet5卷积神经网络
header-img:
tags: CNN
---
# LeNet5介绍
LeNet-5：是Yann LeCun在1998年设计的用于手写数字识别的卷积神经网络，当年美国大多数银行就是用它来识别支票上面的手写数字的，它是早期卷积神经网络中最有代表性的实验系统之一。
LenNet-5共有7层（不包括输入层），每层都包含不同数量的训练参数，如下图所示。 
![](/img/acticle/lenet5_model.png)
LeNet-5中主要有2个卷积层、2个下抽样层（池化层）、3个全连接层3种连接方式

# 构建MNIST识别LeNet5模型
## 输入数据为MNIST_DATA数据
input_tensor.shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUMBER_CHANNELS]
[BATCH_SIZE, 28, 28, 1]

## 卷积层-1
weight = [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP] = [5, 5, 1, 32]
CONV1_SIZE代表卷积过滤器大小
NUM_CHANNELS代表通道数
CONV1_DEEP代表过滤器深度
conv1_biases=[CONV1_DEEP]
输出层结点矩阵为为[BATCH_SIZE, 24, 24, 32]

## 池化层-1
ksize=[1, 2, 2, 1]
strides=[1, 2, 2, 1]
池化层过滤器大小为2，步长为2
则输出层结点矩阵为[BATCH_SIZE, 12, 12, 32]

## 卷积层-2
weights = [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP] = [5, 5, 1, 64]
输出层结点矩阵为[BATCH_SIZE, 8, 8, 64]

## 池化层-2
ksize=[1, 2, 2, 1]
strides=[1, 2, 2, 1]
输出层结点矩阵为[BATCH_SIZE, 4, 4, 64]

## 拉直操作
输出层为[BATCH_SIZE, 16 x 64]

## 全链接-1
输入[BATCH_SIZE, 1024], weight = [1024, 512]

## 全链接-2
输入[BATCH_SIZE, 512], weight = [1024, 10]


```python
import tensorflow as tf

# 配置参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全链接层的结点个数
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP])

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP])

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将矩阵拉直成向量，pool_shape[0]为BATCH_SIZE的大小
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("biases", [FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights), fc1_biases)
        # dropout可以避免过拟合，使得在测试数据上效果更好，一般用于全链接层
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("biases", [NUM_LABELS],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit
```

# LaNet-5的局限性 
CNN能够得出原始图像的有效表征，这使得CNN能够直接从原始像素中，经过极少的预处理，识别视觉上面的规律。然而，由于当时缺乏大规模训练数据，计算机的计算能力也跟不上，LeNet-5 对于复杂问题的处理结果并不理想。

2006年起，人们设计了很多方法，想要克服难以训练深度CNN的困难。其中，最著名的是 Krizhevsky et al.提出了一个经典的CNN 结构，并在图像识别任务上取得了重大突破。其方法的整体框架叫做 AlexNet，与 LeNet-5 类似，但要更加深一些。 