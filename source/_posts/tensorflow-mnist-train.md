---
title: MNIST手写数字识别
toc_nav_num: true
catalog: true
date: 2019-05-11 21:20:13
subtitle: MNIST数字识别
header-img:
tags: tensorflow
---

# MNIST数字识别Code

## tf.train.exponential_decay
学习率指数衰减，参数为：基础学习率,训练次数,batch数量,衰减度

## tf.train.ExponentialMovingAverage
初始化滑动平均类，参数为：滑动平均衰减率，当前训练步数

## tf.control_dependencies
tf.control_dependencies是tensorflow中的一个flow顺序控制机制
作用有二：插入依赖（dependencies）和清空依赖（依赖是op或tensor） 
此处也就是等同于调用tf.group(train_step, variable_average_op)
train_op = tf.no_op(name='train')什么也不做
后面调用train_op的时候，等同于调用train_step, variable_average_op这两个操作都会被调用


```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 输入结点
INPUT_NODE = 784
# 输出结点
OUTPUT_NODE = 10
# 隐藏层书
LAYER1_NODE = 500
# batch包大小
BATCH_SIZE = 100

# 学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化系数
REGULARIZATION_DATE = 0.0001
# 训练轮数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 不使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 不可训练参数
    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # GraphKeys.TRAINING_VARIABLES中的元素，就是可以训练变量的集合
    variable_average_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # tf.argmax(y_, 1)是求y_行中最大值的索引，即为输入标签的数字，y是得出最后的预测实际数字
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_DATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 验证数据
                # 训练过程中验证集采用滑动平均模型得出的accuracy
                validation_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy using average model is %g' % (i, validation_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('./MNIST_DATA', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
```
