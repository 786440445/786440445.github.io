---
title: tensorflow之单层神经网络
catalog: true
date: 2019-05-09 15:45:47
subtitle: 单层神经网络
header-img:
tags: tensorflow
---

# 单层网络
1. x_vals输入的shape=(?, 3),?代表数据集样本个数
1. y_vals输入的shape=(1, ?),?代表数据集样本个数
1. 归一化normalize_cols中axis可以理解为：将矩阵投射到0维度，下每一列选取最大最小值，然后进行归一化处理
1. x_data的shape=(?, 3),?代表一个batch_size大小
1. y_target的shape=(?, 1),?代表一个batch_size大小
1. 隐藏层维度计算为[?, 3] * [3, layer_nodes] + [layer_nodes]
1. 最后模型输出[None, layer_nodes] * [layer_nodes, 1] + [1]

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
# shape = [None, 3]
x_vals = np.array([x[0:3] for x in iris.data])
# shape = [1, None]
y_vals = np.array([x[3] for x in iris.data])

# 设置随机种子
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 归一化处理, axis指的是从第0维度
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 50
x_data = tf.placeholder(tf.float32, shape=[None, 3])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

# 创建模型
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

# 创建隐藏层输出 [None, 3] * [3, layer_nodes] + [layer_nodes]
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
# 创建训练模型的最后输出 [None, layer_nodes] * [layer_nodes, 1] + [1]
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

loss = tf.reduce_mean(tf.square(y_target - final_output))

my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 迭代训练模型
loss_vec = []
test_loss = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # shape = (batch_size, 3)
    rand_x = x_vals_train[rand_index]
    # shape = (batch_size, 1)
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1) % 50 == 0:
        print('Generation: ' + str(i+1) + ' Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
```

![](/img/article/single_layer_nn.png)