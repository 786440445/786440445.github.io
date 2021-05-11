---
title: tensorflow之求逆矩阵以及矩阵分解实现线性回归
catalog: true
toc_nav_num: true
mathjax: true
date: 2019-04-29 21:48:31
subtitle: 线性回归
header-img:
tags: tensorflow
---

# 直接用公式来求解

## 公式原理
线性回归$$Y = w^{T}x + b$$
直接用公式可以解出
A = [X, 1],转换为Y = W*A = [w, b]*[X, 1]
$$w=\left(A^{T} A\right)^{-1} A^{T}y$$
这里求出的w就是[w;b]形式
然后就可以得出直线的斜率和截距

## 求逆矩阵代码
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()

# 数据集合
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# X列向量
x_vals_column = np.transpose(np.matrix(x_vals))
# 列向量，值全为1
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
# A为[X:1], 扩展向量
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))

# 将A向量和b转换为张量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# tA_A为 A的转置乘以A
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
# 求逆运算
tA_A_inv = tf.matrix_inverse(tA_A)
# 乘以转置
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
# 再乘以b
solution = tf.matmul(product, b_tensor)
solution_eval = sess.run(solution)

# 从解中抽取系数,斜率和y截距y-intercept, 即为w和b的值
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope: ' + str(slope))
print('y_intercept' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r--', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()
```

![](/img/article/linearRegess.png)

# 利用tensorflow实现矩阵分解

## 矩阵分解原理
- 这里将用TensorFlow实现矩阵分解，对于一般的线性回归算法，求解Ax=b,则$$x=(A^{T}A)^{-1}A^{T}b$$,然而该方法在大部分情况下是低效率的
- 特别是当矩阵非常大时效率更低。另一种方式则是通过矩阵分解来提高效率。这里采用TensorFlow内建的Cholesky矩阵分解法实现线性回归

1. Cholesky可以将一个矩阵分解为上三角矩阵和下三角矩阵，即$$A = LL^{T}$$
1. 接下来通过tf.cholesky函数计算L,由于该函数返回的是矩阵分解的下三角矩阵，上三角矩阵为该矩阵的转置，L = tf.cholesky(tA_A)
1. $$Ax = b -> A^{T}Ax = A^{T}b -> LL^{T}x = A^{T}b$$
1. tf.matrix_solve()函数返回Ax=b的解。$$Ly = A^{T}b$$ ，求出y
1. $$L^{T}x = y$$，求出x


## 矩阵分解代码
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.Session()
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))
# 将A向量和b转换为张量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# 矩阵分解
# 计算A{T}*A
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tA_A)
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

# 抽取系数
solution_eval = sess.run(sol2)
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slpoe : ' + str(slope))
print('y_intercept : ' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
plt.plot(x_vals, y_vals, '0', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper, left')
plt.show()
```

![](/img/article/LinearR2.png)
