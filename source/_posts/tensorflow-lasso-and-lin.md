---
title: tensorflow之lasso回归和岭回归
catalog: true
date: 2019-04-30 23:13:28
subtitle: lasso回归和岭回归
header-img:
tags: tensorflow
mathjax: true
---

# lasso回归和岭回归
岭回归与Lasso回归的出现是为了解决线性回归出现的过拟合以及在通过正规方程方法求解θ的过程中出现的x转置乘以x不可逆这两类问题的，这两种回归均通过在损失函数中引入正则化项来达到目的，具体三者的损失函数对比见下图： 
线性回归损失函数：
$$
J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$
岭回归损失函数：
$$
J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}
$$
lasso回归损失函数：
$$
J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n}\left|\theta_{j}\right|
$$
其中λ称为正则化参数，如果λ选取过大，会把所有参数θ均最小化，造成欠拟合，
如果λ选取过小，会导致对过拟合问题解决不当，因此λ的选取是一个技术活。 
岭回归与Lasso回归最大的区别在于岭回归引入的是L2范数惩罚项，Lasso回归引入的是L1范数惩罚项，Lasso回归能够使得损失函数中的许多θ均变成0，这点要优于岭回归，因为岭回归是要所有的θ均存在的，这样计算量Lasso回归将远远小于岭回归。 

## lasso
增加损失函数，其为改良过的连续阶跃函数，lasso回归的截止点设为0.9。
Loss = L1_Loss + heavyside_step
$$
heavyside_step = \frac{1}{1+e^{-100(A-0.9)}}
$$
通过这个函数来限制A的值
```python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import datasets

sess = tf.Session()
iris = datasets.load_iris()
# 导入数据
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
# 声明学习率，批量大小，占位符，和模型变量
learning_rate = 0.001
batch_size = 50
x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
# 函数输出值
model_output = tf.add(tf.matmul(x_data, A), b)

##加入L1正则化的损失函数
lasso_param = tf.constant(0.9)
heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, lasso_param)))))
regularization_param = tf.multiply(heavyside_step, 99.)
loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)

# 定义优化器
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i + 1) % 300 == 0:
        print('Step # ' + str(i + 1) + ' A = ' + str(sess.run(A)) +
              ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
```

## 岭回归
对于岭回归在上述代码中修改loss项即可
tf.expand_dims是在后面的式子中加了一个0维度
```python
## 岭回归
ridge_param = tf.constant(1.)
ridge_loss = tf.reduce_mean(tf.square(A))
loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)
```