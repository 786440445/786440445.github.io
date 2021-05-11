---
title: 五层神经网络带L2正则的损失函数计算方法
catalog: true
date: 2019-05-10 15:04:22
subtitle: 神经网络中L2的计算
header-img:
tags: 正则化
---

# 解析
## 集合
tf.add_to_collection('losses', num)
将num收集到collection中，
tf.get_collection可以查看数据，返回是一个列表
tf.add_n(list)将所有数据加起来

## L2正则化
contrib.layers.l2_regularizer(_lambda)(var)
lambda为正则化参数，var为正则化对象


```python
import tensorflow as tf
import tensorflow.contrib as contrib

# 获取一层神经网络的权重，并将权重写入losses集合, _lambda为正则化参数
def get_weight(shape, _lambda):
    var = tf.Variable(tf.random_normal(tf.float32, shape))
    tf.add_to_collection('losses', contrib.layers.l2_regularizer(_lambda)(var))
    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 8
# 定义每层NN的维度
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

# shape = (None, 2)
cur_layer = x
in_dimension = layer_dimension[0]

# 遍历每层网络
for i in range(1, n_layers):
    # 输出层维度
    out_dimension = layer_dimension[1]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    # 使用ReLU激活函数(None, 2) * (2, 10)
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

# 最小均方误差
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 加入损失集合
tf.add_to_collection('losses', mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
```
