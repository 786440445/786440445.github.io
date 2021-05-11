---
title: tensorflow之软支持向量机
catalog: true
date: 2019-05-05 16:12:07
subtitle: 软支持向量机
header-img:
tags: tensorflow
mathjax: true
---

# 软支持向量机原理
## soft margin 损失函数
$$
\frac{1}{n} \sum_{i=1}^{n} \max \left(0,1-y_{i}\left(A x_{i}-b\right)\right)+a\|A\|^{2}
$$
即当样本点在间隔外的时候，损失函数为0，当样本点在间隔内的时候，损失函数为$$1-y_{i}*output$$
$$
output=A x_{i}-b
$$

## 软支持向量机代码

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

# 划分数据集
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 设置批量大小
batch_size = 100
x_data = tf.placeholder(tf.float32, shape=[None, 2])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

# 设置斜率A，和截距b的大小
A = tf.Variable(tf.random.normal(shape=[2, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))

model_output = tf.subtract(tf.matmul(x_data, A), b)

# 声明最大间隔损失函数
l2_norm = tf.reduce_mean(tf.square(A))
# 设置一个aplha值
alpha = tf.constant([0.1])
# 软支持向量机分类误差，即在间隔内部有误差，为1-（输出值*实际值），在间隔外部为0，
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# 声明预测函数和准确度函数
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# 开始遍历迭代训练模型
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

    if (i+1) % 100 == 0:
        print('Step # ' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1
x1_vals = [d[1] for d in x_vals]
best_fit = []
# 按照最优A，b得出直线的点所在的位置,best_fit为第二维的数据
for i in x1_vals:
    best_fit.append(slope * i + y_intercept)

# 求出间隔左右两边的点的集合
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
print(train_accuracy)
print(test_accuracy)
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Accuracies')
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

1. tf.equal(prediction, y_target) 是得到一个true，false的列表
1. tf.cast()是将true，false转换成1或者0
1. 我们每次得到的结果都不尽相同，原因是因为初始化A，b的结果不同，而且每次训练集和测试集的随机分割，每次批量大小不同

![](/img/article/soft-margin-svm01.png)
![](/img/article/soft-margin-svm02.png)
![](/img/article/soft-margin-svm03.png)