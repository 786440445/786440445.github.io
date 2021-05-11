---
title: tensorflow之激励函数和损失函数
toc_nav_num: true
catalog: true
date: 2019-04-26 20:51:42
subtitle: 神经网络
header-img:
tags: tensorflow
mathjax: true
---

# 激励函数

## ReLU
函数为max(0, x)
是神经网络最常用的非线性函数
线形整流单元

## ReLU6
函数为min(max(0, x), 6)
为了抵消ReLU的线形增长部分

## sigmoid
函数为1/(1+exp(-x))
训练过程中反向传播项趋近于0，因此不怎么使用

## tanh
双曲正切函数
与sigmoid类似，不同的是tanh的取值范围是0到1，sigmoid的取值凡事是-1到1
函数是((exp(x)-exp(-x))/(exp(x)+exp(-x)))

## softsign
函数是x/(abs(x)+1), 符号连续估计

## softplus
是ReLU激励函数的平滑版
函数为log(exp(x)+1)

## ELU
与softplus类似
不同点在于当输入无限小时，ELU激励函数趋近于-1，而softplus趋近于0
表达式为(exp(x)+1) if x<0 else x


# 损失函数
![](/img/article/funclost.png)
![](/img/article/funclost2.png)

## L1损失函数
即绝对损失函数

## L2损失函数
```python
	tf.nn.l2_loss()
```

## Pseudo-Huber损失函数
```python
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1) - 1.))
phuber1_y_out = sess.run(phuber1_y_vals)
``` 

## Hinge损失函数
主要用来估计支持向量机算法，但有时也用来估计神经网络算法。

## 两类交叉损失函数（cross-entropy）
作为逻辑损失函数，预测两类目标0或者1时，希望度量预测值到真实分类值（0或1）的距离
```python
xentropy_y_vals = -tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)
``` 

## sigmoid交叉熵损失函数（sigmoid cross entropy）
先把x_vals值通过sigmoid函数转换，再计算交叉熵
```python
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)
``` 

## 加权交叉熵损失函数（Weighted cross-entropy）
```python
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
``` 

## softmax交叉熵损失函数（softmax cross-entropy）
作用于非归一化的输出。通过softmax函数将输出结果转化成概率分布，然后计算真值概率分布损失
```python
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits, target_dist)
``` 

## 稀疏softmax交叉熵损失函数，
和上一个类似，它把目标分类为true转化成index，而softmax交叉熵将目标转成概率分布
```python
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_target_dist, logits=unscaled_logits)
# print(sess.run(sparse_xentropy))
``` 