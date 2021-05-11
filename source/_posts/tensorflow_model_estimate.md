---
title: tensorflow之模型评估
toc_nav_num: true
catalog: true
date: 2019-04-29 21:20:56
subtitle: 回归分类算法评估
header-img:
tags: tensorflow
---

# 回归算法模型评估

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.Session()

# 训练数据
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10, 100)

x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])
batch_size = 25

# 训练集合下标-->随机选取【0-len（x_vals)】中0.8倍数量的下标
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
# 测试集合下标
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
# 随机选取变量A，正太分布中的一个值
A = tf.Variable(tf.random_normal(shape=[1, 1]))

# 声明算法模型,损失函数和优化算法器
my_output = tf.matmul(x_data, A)
# tf.reduce_mean计算batch_size中的平均值
loss = tf.reduce_mean(tf.square(my_output - y_target))
init = tf.initialize_all_variables()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # transpose可以将接收的向量进行转置，需要的是一个列向量[None, 1]，传入的是一个行向量
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1) % 25 == 0:
        print('Step # ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

## 对模型进行评估
mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print('MSE on test : ' + str(np.round(mse_test, 2)))
print('MSE on train : ' + str(np.round(mse_train, 2)))
```

将原数据集一分为2，百分之八十为训练集，百分之二十为测试集
训练集中训练好之后的模型放在测试集进行测试
算出最小均方误差的平均值 tf.reduce_mean(tf.square(my_output - y_target))

# 分类算法模型评估
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
batch_size = 25
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(tf.float32, shape=[1, None])
y_target = tf.placeholder(tf.float32, shape=[1, None])

# 划分训练集测试集
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# 初始化变量,增加模型和损失函数 以及 优化器
my_output = tf.add(x_data, A)
init = tf.initialize_all_variables()
sess.run(init)
# tf.reduce_mean计算batch_size中的平均值
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1) % 200 == 0:
        print('Step # ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))


## 在测试集和训练集中评估训练模型
# tf.squeeze()可以删除维度为1的维度，用squeeze（）函数封装预测操作，使得预测值和目标值有相同的维度。
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
# 返回真值
correct_prediction = tf.equal(y_prediction, y_target)
# 将真假值转换为数字0和1，再计算平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})
print('Accuracy on train set ' + str(acc_value_train))
print('Accuracy on test set ' + str(acc_value_test))


# 画图
A_result = sess.run(A)
bins = np.linspace(-5, 5, 50)
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color='blue')
plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red')
plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = ' + str(np.round(A_result, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy=' + str(np.round(acc_value_test, 2)))
plt.show()
```

将原数据集一分为2，百分之八十为训练集，百分之二十为测试集
训练集中训练好之后的模型放在测试集进行测试
分类算法中是计算分类的正确率
损失函数为sigmoid的交叉熵

![](/img/article/guji.png)