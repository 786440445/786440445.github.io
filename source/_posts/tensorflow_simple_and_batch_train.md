---
title: tensorflow之随机训练和批量训练
toc_nav_num: true
catalog: true
date: 2019-04-27 15:37:39
subtitle: 神经网络
header-img:
tags: tensorflow
---

# 随机训练与批量训练
![](/img/article/suijixunlian.png)

随机训练损失更不规则，批量训练更平滑

```python
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
sess = tf.Session()

# 设置批量大小
batch_size = 20

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[1, 1]))

# 目标输出函数
my_output = tf.matmul(x_data, A)

# 初始化A
init = tf.initialize_all_variables()
sess.run(init)

# 计算每个数据点的L2损失的平均值
loss = tf.reduce_mean(tf.square(my_output - y_target))

# 定义梯度下降优化器
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# 批量训练
loss_batch = []
for i in range(100):
    # 定义的批大小为20个，即生成一个20个x_val的行向量
    # 生成20个y_val的行向量
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1) % 5 == 0:
        print('Step1 #' + str(i+1) + 'A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss1 = ' + str(temp_loss))
        loss_batch.append(temp_loss)

# 随机训练
# 重新选定A值
A = tf.Variable(tf.random_normal(shape=[1, 1]))
init = tf.initialize_all_variables()
sess.run(init)

loss_stochastic = []
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [[x_vals[rand_index]]]
    rand_y = [[y_vals[rand_index]]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1) % 5 == 0:
        print('Step2 #' + str(i + 1) + 'A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss2 = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)

# 绘图
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='StochasticLoss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='BatchLoss size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()
```