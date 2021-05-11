---
title: tensorflow再次实现简单的CNN
catalog: true
date: 2019-06-06 22:24:20
subtitle: 简单的CNN
header-img:
tags: tensorflow
---

# 简单的CNN
1. padding='VALID'指卷积边界无填充，'SAME'是指卷积边界有填充，填充到原大小
1. 设置keep_prob=0.5的进行训练，即减少部分网络神经防止过拟合，然后测试准确率使用全部网络结构
1. reduction_indices指reduce_mean的维度方向

```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('../Chapter4/MNIST_data/', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_varibale(shape):
    initail = tf.constant(0.1, shape=shape)
    return tf.Variable(initail)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 四维向量输入
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 第一个卷积层：卷积核大小，输入通道数为1，输出通道数为32，输出宽度为[28, 28]
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_varibale([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# 池化层输出宽度为[14, 14]
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷句层：卷积核大小，输入通道为32，输出通道为64,输出宽度为[14, 14]
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_varibale([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# 池化层输出宽度为[7, 7]
h_pool2 = max_pool_2x2(h_conv2)

# 全链接层1,全链接输出1024个结点
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_varibale([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)
# 全链接层2,输出10个结点
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_varibale([10])
y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, w_fc2) + b_fc2)

# 计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step: %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}))
```