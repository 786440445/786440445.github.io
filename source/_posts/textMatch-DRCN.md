---
title: 密集循环注意力网络DRCN
catalog: true
date: 2020-04-09 11:24:23
subtitle: 深度学习
header-img:
tags: 文本匹配
---

## DRCN：密集循环注意力网络

### 1.Word Representation Layer

​	将输入的两个句子表示为P和Q，长度分别为I和J。每个词的word representation feature由三部分构成：word embedding, character representation, the exact matched flag。

1. 其中word embedding包括固定好的预训练模型（GloVe或者Word2Vec）和可以训练的WordEmbedding。可训练的wordembedding会导致过拟合。固定的wordembedding缺少特殊场景的数据会很健壮。融合选取。
2. Character Embedding是随着模型训练的。
3. the exact matched flag则表示代表词是否在其他句子中出现。类似DIIN中的EM标志。

### 2.Attentively Connected RNN

#### 2.1Densely-connected Recurrent Networks

​	传统的多层RNN结构在层数加深的过程中会遭遇梯度消失或梯度爆炸等问题。为了解决此问题，研究者们提出了残差网络ResNet，即在网络前向传播中增加跳跃传播的过程，将信息更有效地传播至更深的层。

​	但是ResNet的做法是将输入与输出相加作为下一层的输入。这样输入的信息会在传递的过程中产生一些影响。

​	借用DenseNet将输入和输出串联起来。通道数维度上进行连接。很好的进行传递。

#### 2.2Densely-connected Co-attentive Networks

​	Attention机制可以在两个句子间建立对应关系，在多层RNN的每层中计算P和Q的attention信息，通过串联拼接得到每层RNN的输出。

![v2-a3baf1e6e428303b7414d0eb0c927c74_hd](\img\article\v2-a3baf1e6e428303b7414d0eb0c927c74_hd.jpg)

![v2-642056e4a761b6487c768e33ffe3cf1b_hd](\img\article\v2-642056e4a761b6487c768e33ffe3cf1b_hd.jpg)

#### 2.3Bottleneck Component

​	由于在多层RNN中进行了拼接操作，随着层数的加深，需要训练的参数会大量增加，因此引入自编码模块进行特征维数的压缩，实验证明，此操作对测试集起到了正则作用，提高了测试准确率。

### 3.Interaction and Prediction Layer

​	由于P、Q的长度不同，本文通过Max Pooling对每个句子分别得到其向量表示p、q，计算得到含有语义匹配信息的向量v，具体如下：

![v2-fe65e454c2db3aeb00a2b911f2f4b729_hd](\img\article\v2-fe65e454c2db3aeb00a2b911f2f4b729_hd.jpg)

最后将v输入全连接层，并根据具体任务通过softmax进行分类。

本文整体架构是一个端到端的过程，损失函数定义为交叉熵损失与自编码器的重建损失之和，并选用RMSProp算法进行参数优化。