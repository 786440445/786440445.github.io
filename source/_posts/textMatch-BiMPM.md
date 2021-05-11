---
title: 双向多视角匹配BiMPM
catalog: true
date: 2020-04-09 11:27:32
subtitle:
header-img:
tags: 文本匹配
---

## BiMPM：双向多视角匹配

​	文本蕴含或者自然语言推理任务，就是判断后一句话（假设句）能否从前一句话（前提句）中推断出来。

​	在BiMPM论文提出前，大量的学者在对两句话进行匹配时，常常仅考虑到前一句话对后一句话，或后一句话对前一句话的单向语义匹配，忽略了双向语义匹配的重要性；

​	并且一般只考虑单一粒度的语义匹配（逐字或逐句）。基于以上不足，该论文作者提出了bilateral multi-perspective matching （BiMPM）model，即双向多视角匹配模型。

### 2.Context Representation Layer

​	将上下文信息融入到句子P和句子Q的每一个time step中。首先用一个双向LSTM编码句子p中每一个time step的上下文embedding。

​	使用同一个双向LSTM，编码q中的每一个time step的上下文embedding。即将该双向LSTM进行权值共享。

### 3.Matching Layer

​	这一层是本模型的核心，也是亮点。本层的目的是用一句话中每一个time step的上下文向量去匹配另一句话中所有time steps的上下文向量。如图1所示，本模型从两个方向（P->Q和Q->P）去匹配句子P和句子Q的上下文向量。下面仅从一个方向P->Q，详细讲一下多视角匹配算法，另一方向Q->P与其相同。

多视角匹配算法包含两步：

（1）定义了多视角余弦匹配函数 fm去比较两个向量，即 ![[公式]](https://www.zhihu.com/equation?tex=m%3Df_%7Bm%7D%28v_%7B1%7D%2Cv_%7B2%7D%3BW%29) 

其中v1和v2是d维的向量。W（lxd）是一个可训练的权重参数，l是视角的数目，也就是共有几个视角（可以理解成CNN做卷积时的多个filters）。返回值m是l维的向量![[公式]](https://www.zhihu.com/equation?tex=m%3D%5Bm_%7B1%7D%2C...m_%7Bk%7D%2C...m_%7Bl%7D%5D)。

其中mk是第k个视角的向量余弦匹配值，即 ![[公式]](https://www.zhihu.com/equation?tex=m_%7Bk%7D%3Dcosine%28W_%7Bk%7D%5Ccirc+v_%7B1%7D%2CW_%7Bk%7D%5Ccirc+v_%7B2%7D%29) 

其中wk是第k行W的值

（2）基于 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bm%7D) 函数，本模型给出了四种匹配策略，分别是full-matching、maxpooling-matching、attentive-matching和max-attentive-matchong，如图2所示。

![v2-6f5d775be674bfe4a73ac0f7c10ccae2_r](\img\article\v2-6f5d775be674bfe4a73ac0f7c10ccae2_r.jpg)

#### 1) full-matching

​	在这中匹配策略中，我们将句子P中每一个time step的上下文向量（包含向前和向后上下文向量）分别与句子Q中最后一个time step的上下文向量（向前和向后上下文向量）计算余弦匹配值

![v2-7d83edaa4ad519d991e6d15a3484f227_720w](\img\article\v2-7d83edaa4ad519d991e6d15a3484f227_720w.jpg)

#### 2) maxpooling-matching

​	我们将句子P中每一个time step的上下文向量（包含向前和向后上下文向量）分别与句子Q中每一个time step的上下文向量（向前和向后上下文向量）计算余弦匹配值，但最后在与句子Q的每一个time step中选取最大的余弦匹配值

![v2-abb4b0e0f6ea3d382c11a251d30ede44_r](\img\article\v2-abb4b0e0f6ea3d382c11a251d30ede44_r.jpg)

#### 3) attentive-matching

​	我们先对句子P和句子Q中每一个time step的上下文向量（包含向前和向后上下文向量）计算余弦相似度（这里值得注意的一点是，余弦匹配值与余弦相似度是不一样的，余弦匹配值在计算时对两个向量赋予了权重值，而余弦相似度则是直接对两个向量进行计算），得到相似度矩阵。

![v2-7fa2c277fd1a5a3f52b674ddd43df362_r](\img\article\v2-7fa2c277fd1a5a3f52b674ddd43df362_r.jpg)

​	我们将相似度矩阵，作为句子Q中每一个time step的权值，然后通过对句子Q的所有上下文向量加权求和，计算出整个句子Q的注意力向量。

![v2-2213b5e3fdecd1d78a0cae64518b2cf5_720w](\img\article\v2-2213b5e3fdecd1d78a0cae64518b2cf5_720w.jpg)

最后，将句子P中每一个time step的上下文向量（包含向前和向后上下文向量）分别与句子Q的注意力向量计算余弦匹配值，即

![v2-9e2ef86e2f00c986d10802ffe94fb4c1_720w](\img\article\v2-9e2ef86e2f00c986d10802ffe94fb4c1_720w.jpg)

#### 4) max-attentive-matching

​	这种匹配策略与attentive-matching的匹配策略相似，不同的是，该匹配策略没有对句子Q的所有上下文向量加权求和来得到句子Q的注意力向量，而是选择句子Q所有上下文向量中余弦相似度最大的向量作为句子Q的注意力向量。

### 4. aggregation Layer

​	这一层的目的是将两个序列的匹配向量聚合成一个固定长度的匹配向量。本模型利用另一个双向LSTM，将其分别应用于两个序列的匹配向量。然后，通过将双向LSTM的最后一个time step的向量串联起来（图2中四个绿色的向量），聚合成固定长度的匹配向量。

### 5. Prediction Layer

​	这一层的目的是为了得到最终的预测结果。本模型将聚合得到的匹配向量，连接两层全连接层，并且在最后输出做softmax激活，最后得到文本蕴含的结果。

### 6. 参数

​	该论文中，word embedding为300维，character embedding为20维，得到的character-composed embedding为50维；所有的双向LSTM的隐层节点数为100，dropout rate为0.1，学习率为0.001，采用adam优化器。论文中，对五种（1, 5, 10, 15, 20）不同大小的视角进行实验分析，发现视角个数为20时，模型效果最好