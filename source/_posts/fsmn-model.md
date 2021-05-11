---
title: FSMN前馈型序列记忆网络
catalog: true
date: 2019-07-10 19:57:07
subtitle: 科大讯飞模型
header-img:
tags: 语音识别
mathjax: true
---

# FSMN(前馈型序列记忆网络)

## 模型结构
![](/img/article/fsmn_1.png)
![](/img/article/fsmn_2.png)

## 模型介绍
1. 在隐藏层的旁边，FSMN挂了一个记忆模块Memory Block，记忆模块的作用与LSTM门结构类似，可以用来记住t时刻输入信息的相邻时刻序列的信息。
1. 根据记忆模块编码方式的区别，FSMN又可以分为sFSMN和vFSMN，前者代表以标量系数编码，后者代表以向量系数编码。

## 公式推导
- 如上图，以记住前N个时刻信息为例子，计算公式如下
$$
\overrightarrow{\tilde{h}}_{t}^{l}=\sum_{i=0}^{N} a_{i}^{l} \cdot \overrightarrow{h_{t-i}^{l}}, \text {in... sFSMN}
$$
$$
\overrightarrow{\tilde{h}}_{t}^{l}=\sum_{i=0}^{N} \overrightarrow{a_{i}^{l}} \odot \overrightarrow{h_{t-i}^{l}}, \text {in...} v F S M N
$$
- 一式表示标量乘积，二式表示Hadamard积
- 有了这个隐藏层旁挂着的记忆模块，就要将此记忆模块作为输入传递到下一个隐藏层
$$
h_{t}^{\overrightarrow{l+1}}=f\left(W^{l} \overrightarrow{h_{t}^{l}}+\tilde{W}^{l} \overrightarrow{\tilde{h}_{t}^{l}}+\overrightarrow{b^{l}}\right)
$$
- 以上就是简单的回看式FSMN，也就是说当下的记忆模块只关注了它之前的信息，如果还要关注未来的信息，实现上下文联通，也就是所谓的双向的FSMN，直接在上式中添加后看的阶数即可，如下：
![](/img/article/fsmn_3.png)
其中N1和N2分别代表前看和后看的阶数。

## 简而言之
给定一个包含T个单词的序列X，我们可以构造一个T阶的方阵M：
![](/img/article/fsmn_4.png)
![](/img/article/fsmn_5.png)
鉴于上式，我们就有了很美的以下这个公式：
$$\tilde{H}=H M$$
更为推广的，对于给定的K个序列：
$$
L=\left\{X_{1}, X_{2}, \ldots, X_{K}\right\}
$$
一个更美的公式诞生了：
$$
\tilde{H}=\left[H_{1}, H_{2}, \ldots, H_{K}\right]\left[\begin{array}{cccc}{M_{1}} & {} & {} & {} \\ {} & {M_{2}} & {} & {} \\ {} & {} & {\ddots} & {} \\ {} & {} & {} & {M_{K}}\end{array}\right]=\overline{H} \overline{M}
$$

