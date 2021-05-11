---
title: 最佳实战MNIST识别
toc_nav_num: true
catalog: true
date: 2019-05-17 15:29:02
subtitle: tensorflow实战mnist识别
header-img:
tags: tensorflow
---

# 神经网络架构
mnist_inference.py
```python
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 得到中间权重值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights

# 构建两层网络
def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
```

# 训练网络
1. 这里每训练一千次就将sess进行保存

mnist_train.py
```python
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import Google_Tensorflow.part_5.mnist_inference as mnist_inference

# batch大小
BATCH_SIZE = 100
# 基础学习率
LEARNING_RATE_BASE = 0.8
# 学习率延迟
LEARNING_RATE_DECAY = 0.99
# 正则化参数
REGULARAZTION_RATE = 0.0001
# 训练步数
TRAINING_STEPS = 30000
# 滑动平均延迟
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径和文件名
MODEL_SAVE_PATH = "./model1/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    # 正则化参数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    # 滑动平均值
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('./MNIST_DATA', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
```

# 测试训练
ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
每次会读取最新保存的模型，并验证其正确率
```python
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import Google_Tensorflow.part_5.mnist_inference as mnist_inference
import Google_Tensorflow.part_5.mnist_train as mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 计算准确率
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)

        #
        validates_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(validates_to_restore)

        while True:
            with tf.Session() as sess:
                # 检查点状态，checkpoint文件
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 从当前检查点载入sess，并运行得到准确率
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets('./MNIST_DATA', one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
```