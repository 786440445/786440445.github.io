---
title: tensorflow之实现多分类SVM
catalog: true
date: 2019-05-06 21:15:34
subtitle: 多分类SVM
header-img:
tags: tensorflow
---

# 多分类SVM
## 原理
通过参数b增加一个维度计算三个模型
使用一对多，为每类创建一个分类器，最后预测类别是具有最大SVM间隔的分类

## code
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals1 = np.array([1 if y == 0 else -1 for y in iris.target])
y_vals2 = np.array([1 if y == 1 else -1 for y in iris.target])
y_vals3 = np.array([1 if y == 2 else -1 for y in iris.target])
y_vals = np.array([y_vals1, y_vals2, y_vals3])

class1_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 0]
class1_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 0]
class2_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 1]
class2_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 1]
class3_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 2]
class3_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 2]

# 数据集维度在变化，从单目标分类到三类目标分类。
# 我们将利用矩阵传播和reshape技术一次性计算所有的三类SVM
batch_size = 50
x_data = tf.placeholder(tf.float32, shape=[None, 2])
y_target = tf.placeholder(tf.float32, shape=[3, None])
prediction_grid = tf.placeholder(tf.float32, shape=[None, 2])

b = tf.Variable(tf.random_normal(shape=[3, batch_size]))

# 计算高斯核函数
# 声明高斯核函数
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
# 实现了(xi-xj)的平方项
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [3, batch_size, 1])
    return(tf.matmul(v2, v1))

# 计算对偶问题
model_output = tf.matmul(b, my_kernel)
# 损失函数对偶问题的第一项
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)
# 损失函数对偶问题的第二项
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
# 第一项加第二项的负数
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# 创建预测函数和准确度函数,先创建一个预测核函数
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
# (x_data - prediction_grid)的平方项
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# 预测输出
# 实现预测核函数后，我们创建预测函数。
# 与二类不同的是，不再对模型输出进行sign（）运算。
# 因为这里实现的是一对多方法，所以预测值是分类器有最大返回值的类别。
# 使用TensorFlow的内建函数argmax（）来实现该功能
prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
# 计算每一个预测的平均值
prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

# 声明优化器
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
batch_accuracy = []
for i in range(100):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:, rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)

    if (i+1) % 25 == 0:
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
grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# # 绘图
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.get_cmap('Paired'), alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='I setosa')
plt.plot(class2_x, class2_y, 'kx', label='I versicolor')
plt.plot(class3_x, class3_y, 'gv', label='I virginica')
plt.title('Gaussian SVM Result on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3])
plt.xlim([3.5, 8.5])
plt.show()

plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
```

![](/img/article/multy_classfy_svm01.png)
![](/img/article/multy_classfy_svm02.png)
![](/img/article/multy_classfy_svm03.png)