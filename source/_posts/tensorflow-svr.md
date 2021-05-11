---
title: tensorflow之支持向量回归
catalog: true
date: 2019-05-05 17:15:33
subtitle: SVR
header-img:
tags: tensorflow
mathjax: true
---

# SVR
## 损失函数
$$
\max \left(0,\left|y_{i}-\left(A x_{i}+b\right)\right|-\varepsilon\right)
$$

## code
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# 划分数据集
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 设置批量大小
batch_size = 50
x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

# 设置斜率A，和截距b的大小
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))

model_output = tf.add(tf.matmul(x_data, A), b)

# 声明损失函数、
epsilon = tf.constant([0.5])
loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))

my_opt = tf.train.GradientDescentOptimizer(0.075)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

train_loss = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_train_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
    train_loss.append(temp_train_loss)
    temp_test_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
    test_loss.append(temp_test_loss)

    if (i+1) % 50 == 0:
        print('------------')
        print('Generation : ' + str(i))
        print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Train Loss = ' + str(temp_train_loss))
        print('Test Loss = ' + str(temp_test_loss))

[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)

best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
    best_fit_upper.append(slope * i + y_intercept + width)
    best_fit_lower.append(slope * i + y_intercept - width)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(train_loss, 'k-', label='Train Set Loss')
plt.plot(test_loss, 'r--', label='Test Set Loss')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.legend(loc='upper right')
plt.show()
```

## something need to say
支持向量回归中，SVR需要我们定义一个常量ϵ>0,对于某一个点(xi,yi)，如果|yi−w∙ϕ(xi)−b|≤ϵ，则完全没有损失;如果|yi−w∙ϕ(xi)−b|>ϵ,则对应的损失为|yi−w∙ϕ(xi)−b|−ϵ
这个均方差损失函数不同，如果是均方差，那么只要yi−w∙ϕ(xi)−b≠0，那么就会有损失。

![](/img/article/svr01.png)
![](/img/article/svr02.png)