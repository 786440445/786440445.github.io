---
title: tensorflow实现自编码器
catalog: true
toc_nav_num: true
date: 2019-06-06 13:18:05
subtitle: 自编码器
header-img:
tags: tensorflow
mathjax: true
---

# 自编码器
1. 最具代表性质的去燥编码器
1. xavier初始化使根据某一层网络的输入，输出结点数量自动调整最合适的分布，使均值为0，方差为2/(n_in+n_out)
1. standard_scale是对数据进行归一化处理，即均值为0，方差为1
1. softplus是ReLU激励函数的平滑版，函数为$$
\zeta(x)=\log \left(1+e^{x}\right)
$$ 
		

# 代码
```python
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data


# 数据预处理，使权重初始化不大不小，即均方为0，方差为2/(n_in+n_out)
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# 实现一个自编码器
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        # 隐藏层激活函数
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weight = self._initialize_weights()
        self.weights = network_weight
        # 输入x，结点数为n_input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 隐藏层，存在一个噪音百分比scale
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        # 输出层
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 平方误差
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 初始化参数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 用一个batch数据训练，并返回cost
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 计算总的误差cost
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 输出隐藏层数据
    def transfer(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 输出输出层结果
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 输出层结果
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    # 权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 偏移量
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 归一化处理数据，均值为0方差为1
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 随机生层一个batch数据块
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

auto_encoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                                n_hidden=200,
                                                transfer_function=tf.nn.softplus,
                                                optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = auto_encoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
print("Total cost: " + str(auto_encoder.calc_total_cost(X_test)))
```

# 结果
![](/img/article/auto_encoder.png)