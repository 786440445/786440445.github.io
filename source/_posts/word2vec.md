---
title: word2vec
catalog: true
date: 2020-04-22 21:55:13
subtitle: 词向量
header-img:
tags: 自然语言处理
---

# Word2vec

### 一. wordvec简介

1. wordvec是一种训练词向量分布式表示（distrubute representation）的方式。
2. 分为cbow和skip-gram两种训练方式。cbow通过上下文词汇作为输入，中间词汇作为输出。skip-gram通过中间词汇作为输入，上下文词汇作为输出，进行训练。
   - 如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『Skip-gram 模型』。适合大规模数据集中。
   - 而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『CBOW 模型』。适合少量数据集中。
3. 通常的文本数据中，词库少则数万，多则百万，在训练中直接训练多分类逻辑回归并不现实。word2vec中提供了两种针对大规模多分类问题的优化手段， negative sampling（负采样）和hierarchical softmax（层次softmax）。

### 二. 负采样

1. 在优化中，negative sampling 只更新少量负面类，从而减轻了计算量。hierarchical softmax 将词库表示成前缀树，从树根到叶子的路径可以表示为一系列二分类器，一次多分类计算的复杂度从|V|降低到了树的高度。

![20190319100122167](\img\article\20190319100122167.png)

2. 负采样，negative sampling。是指在skip-gram过程中。将中间词汇和上下文正确匹配的词汇形成正例。而把匹配失败的样本称为负例子。当训练样本过大时候，负样本数量是远远大于正样本数量的。因此如果使用p（w1|w2）常规的方法来计算损失函数，时间复杂度会很大。

3. 采用负采样技术，在负样本中随机选择部分进行损失函数的计算。

4. word2vec源代码中使用tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes)。range_max=num_classes。
5. 默认情况下，他会用log_uniform_candidate_sampler去采样。那么log_uniform_candidate_sampler是怎么采样的呢？他的实现在这里：1、会在[0, range_max)中采样出一个整数k。2、P(k) = (log(k + 2) - log(k + 1)) / log(range_max + 1)
6. TF的word2vec实现里，词频越大，词的类别编号也就越小。因此，在TF的word2vec里，负采样的过程其实就是优先采词频高的词作为负样本。

###  三. 层次softmax

​	在层次softmax模型中，叶子结点的词没有直接输出的向量，而非叶子节点都有响应的输在在模型的训练过程中，通过Huffman编码，构造了一颗庞大的Huffman树，同时会给非叶子结点赋予向量。我们要计算的是目标词w的概率，这个概率的具体含义，是指从root结点开始随机走，走到目标词w的概率。因此在途中路过非叶子结点（包括root）时，需要分别知道往左走和往右走的概率。例如到达非叶子节点n的时候往左边走和往右边走的概率分别是：

![20190319100307838](\img\article\20190319100307838.png)