---
title: tensorflow之实现线性回归
toc_nav_num: true
catalog: true
date: 2019-04-30 20:24:04
subtitle: 线性回归
header-img:
tags: tensorflow
mathjax: true
---

# 实现线性回归
## LinearRegression原理
采用最小二乘法求所有样本点的的$$y_{i}$$到Y的总距离作为损失函数，然后求其最小值

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
iris = datasets.load_iris()
# 导入数据
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# 声明学习率，批量大小，占位符，和模型变量
learning_rate = 0.05
batch_size = 25
x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 函数输出值
model_output = tf.add(tf.matmul(x_data, A), b)

# 声明L2损失函数, 定义优化器，然后进行优化
loss = tf.reduce_mean(tf.square(y_target - model_output))
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# 遍历迭代计算
loss_vec = []
for i in range(100):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1) % 25 == 0:
        print('Step # ' + str(i+1) + ' A = ' + str(sess.run(A)) +
              ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

[slope] = sess.run(A)
[y_intecept] = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intecept)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()
```

![](/img/article/Linear_03.png)
![](/img/article/Linear_04.png)

# 实现戴明回归
## DemingRegression原理
给定直线为$$y=mx+b$$, 点$$(x_{0},y_{0})$$,求两者间的距离公式为
$$
d=\frac{\left|y_{0}-\left(m x_{0}+b\right)\right|}{\sqrt{m^{2}+1}}
$$
所有的d总距离作为损失函数，并求其最小值

```python
# 函数输出值
model_output = tf.add(tf.matmul(x_data, A), b)
# 计算点到直线的距离总和为损失函数, 定义优化器，然后进行优化
# 计算|y-(m*x + b)|
deming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
# 计算(m^2 + 1)^ 1/2，+1是为了防止斜率m为0的情况
deming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
# 上式两者相除即为点到直线的距离
loss = tf.reduce_mean(tf.truediv(deming_numerator, deming_denominator))
```

![](/img/article/Linear_05.png)
![](/img/article/Linear_06.png)
