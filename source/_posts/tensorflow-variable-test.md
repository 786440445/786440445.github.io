---
title: tensorflow变量管理
catalog: true
toc_nav_num: true
date: 2019-05-13 21:59:43
subtitle: 变量管理
header-img:
tags: tensorflow
---

# 变量命名空间
```python
import tensorflow as tf

# 等价定义
a = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
b = tf.Variable(tf.constant(1.0, shape=[1], name='v'))


# 在名字为foo的命令空间创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间foo中已经存在名字为v的变量，所有下面的代码将会报错
with tf.variable_scope("foo"):
    v1 = tf.get_variable("v", [1])

with tf.variable_scope("foo", reuse=True):
    v2 = tf.get_variable("v", [1])
    print(v == v2)  # True

# 将参数设置为True时，tf.get_variable只能获取已经创建过的变量，
# 命名空间bar中没有创建变量v，所以下面会报错
with tf.variable_scope("bar", reuse=True):
    v = tf.get_variable("v", [1])
```


# 两层网络变量处理
```python
def inference(input_tensor, resue=False):
    with tf.variable_scope('layer1', resue=resue):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', resue=resue):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2
```

# 命名空间嵌套
```python
v = tf.get_variable("v", [1])
with tf.variable_scope("root"):
    v1 = tf.get_variable("v", [1])
    print(v1.name)
    print(tf.get_variable_scope().reuse)  # 输出False
    with tf.variable_scope("foo"):
        v2 = tf.get_variable("v", [1])
        print(v2.name)
        print(tf.get_variable_scope().reuse)  # 输出False
        with tf.variable_scope("bar", reuse=True):
            print(tf.get_variable_scope().reuse)  # 输出True
    print(tf.get_variable_scope().reuse)  # 输出False，退出
```