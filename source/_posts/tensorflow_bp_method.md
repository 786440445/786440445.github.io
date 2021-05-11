---
title: tensorflow之实现反向传播
toc_nav_num: true
catalog: true
date: 2019-04-26 23:00:46
subtitle: 神经网络
header-img:
tags: tensorflow
---

# 工作原理
1. 生成数据
2. 初始化占位符和变量
3. 创建损失函数
4. 定义一个优化器Optimizer
5. 通过随机样本进行迭代，更新变量

# 回归算法实例

```python
import numpy as np
import tensorflow as tf

sess = tf.Session()

# 生成数据 x为正太分布，y为输出值
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10, 100)

x_data = tf.placeholder(tf.float32, shape=[1])
y_target = tf.placeholder(tf.float32, shape=[1])
A = tf.Variable(tf.random_normal(shape=[1]))

# 增加乘法操作
my_output = tf.multiply(x_data, A)

# 增加L2正则损失函数
loss = tf.square(my_output - y_target)

# 运行之前初始化变量
init = tf.initialize_all_variables()
sess.run(init)

# 声明优化器
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

# 训练算法 迭代100步
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1) % 25 == 0:
        print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
```

1. 生成输入数据x_vals，为N(1, 0.1)的正太分布，100个数据量
2. y_vals = 全部为10
3. 随机生成正太分布变量A
4. 目标函数为 A*X
5. 采用L2正则损失函数，最小化目标函数的损失函数
6. 使用梯度下降，学习率为0.02
7. 迭代100次，随机在x_vals，y_vals中选取x,进行优化
8. 输出A的值以及loss

# 二值分类算法
```python
from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf

ops.reset_default_graph()
sess = tf.Session()

# 从正太分布(N(-1,1), N(3,1))生成数据，同时也生成目标标签
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(tf.float32, shape=[1])
y_target = tf.placeholder(tf.float32, shape=[1])
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# 增加转换操作
my_output = tf.add(x_data, A)
# 增加一个维度
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# 初始化A
init = tf.initialize_all_variables()
sess.run(init)

# 声明损失函数
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target_expanded, logits=my_output_expanded)

# 增加优化器
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# 随机选取数据迭代几百次，相应的更新变量A
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1) % 200 == 0:
        print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)))
        print('Loss #' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

```

1. 生成x_vals100个样本，其中一半是N(-1,1)分布，一半是N(3,1)分布
2. 生成y_vals100个样本，一半是0，一半是1
3. 随机生成N(10, _)的变量
4. 目标函数为sigmoid(A+X)
5. 目标A的值为1，因为-1的分类为0，3的分类为1，中间值为1，则1+A=0，A=-1
6. 损失函数为sigmoid交叉熵损失函数
7. 梯度下降优化，步长为0.05
8. 迭代1400次随机选取x,y，输出A和损失函数


