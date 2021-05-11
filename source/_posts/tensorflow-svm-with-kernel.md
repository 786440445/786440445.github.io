---
title: tensorflow之核函数SVM
catalog: true
date: 2019-05-06 09:25:52
subtitle: 高斯核SVM
header-img:
tags: tensorflow
mathjax: true
toc_nav_num: true
---

# 带有核函数的SVM
## 带有核函数的对偶问题
$$
\max \sum_{i}^{m} a_{i}-\frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{m} a_{i} a_{j} y_{i} y_{j} K(x_{i} x_{j})
$$
$$
st :  \sum_{\text {i}}^{m} a_{i} y_{i}=0
$$
$$
a_{i} \geqslant 0, i=1,2, \ldots, m
$$

## 常用核函数
线性核函数
$$
k\left(x_{i}, x_{j}\right)=x_{i}^{T} x_{j}
$$
多项式核函数(d>=1为多项式次数)
$$
k\left(x_{i}, x_{j}\right)=\left(x_{i}^{T} x_{j}\right)^{d}
$$
高斯核函数($$\sigma>0$$为高斯核的带宽)
$$
k\left(x_{i}, x_{j}\right)=\exp \left(-\frac{\left\|x_{i}-x_{j}\right\|^{2}}{2\sigma^{2}}\right)
$$
拉普拉斯核函数($$\sigma>0$$)
$$
k\left(x_{i}, x_{j}\right)=\exp \left(-\frac{\left\|x_{i}-x_{j}\right\|}{\sigma}\right)
$$
Sigmoid核函数(tanh为双曲正切函数,$$\beta>0, \theta<0$$)
$$
k\left(x_{i}, x_{j}\right)=\tanh \left(\beta x_{i}^{\top} x_{j}+\theta\right)
$$

## code review
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
# 生成环形数据
# n_samples：控制样本点总数
# noise：控制属于同一个圈的样本点附加的漂移程度
# factor：控制内外圈的接近程度，越大越接近，上限为1
(x_vals, y_vals) = datasets.make_circles(n_samples=500, factor=.5, noise=.1)

y_vals = np.array([1 if y == 1 else -1 for y in y_vals])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

# 声明批量大小
batch_size = 250
# 样本点的数据x为一个二维数据
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# 样本点的数据y为一个1或者-1的数据
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# 彩色网格可视化不同的区域
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
b = tf.Variable(tf.random.normal(shape=[1, batch_size]))

# 创建高斯核函数
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
# 实现了(xi-xj)的平方项
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# 处理对偶问题
model_output = tf.matmul(b, my_kernel)
# 损失函数对偶问题的第一项
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
# 损失函数对偶问题的第二项
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
# 第一项加第二项的负数
loss = tf.negative(tf.subtract(first_term, second_term))

# 创建预测函数和准确度函数,先创建一个预测核函数
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
# (x_data - prediction_grid)的平方项
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# 预测输出
prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# 创建优化器函数
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 开始迭代训练
loss_vec = []
batch_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)

    if (i+1) % 100 == 0:
        print('Step # ' + str(i+1))
        print('Loss = ' + str(temp_loss))

# 得到第一列里的最小值，和最大值
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
# 得到第二列里的最小值，和最大值
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1

# 步长为0.02均分x_min-x_max形成一个向量
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
# # 将两个xx，yy向量 拼成一个矩阵
grid_points = np.c_[xx.ravel(), yy.ravel()]

[grid_predicttions] = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: grid_points})
grid_predicttions = grid_predicttions.reshape(xx.shape)
print(grid_predicttions)
# # 绘图
plt.contourf(xx, yy, grid_predicttions, cmap=plt.cm.get_cmap('Paired'), alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.plot('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
```

1. dist = tf.reduce_sum(tf.square(x_data), 1)， dist = tf.reshape(dist, [-1, 1])是为了求向量每一个值的平方和
1. tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))是为了求$$||(x_{i}-x_{j})||^2$$
1. 将高斯核替换成其他的核即可实现其他的核SVM