---
title: kaldi
catalog: true
toc_nav_num: true
mathjax: true
date: 2021-05-01 01:57:18
subtitle: speech
header-img:
tags: ASR
---


## 生成模型和判别模型

1. 生成模型：基于观测值x，求其隐藏状态值y的过程。其中x是由o以某种概率密度分布进行决定。主体是y，因此p(y|x)无法根据条件概率直接建模。需要根据贝叶斯公式进行转化。
$$
p(y|x)=\frac{p(x,y)}{p(x)}=\frac{p(x|y)p(y)}{p(x)}
$$
    
> 生成模型包括有：GMM，HMM，Biays网络

- 其中P(x)是先验知识，可以直接计算出来。因此该式转换为P(x|y)来计算。

2. 判别模型：基于观测值x，求y的条件概率p(y|x)，主体是x，y由x进行决定。可以直接对观测值，建模，推测出Y的结果。

> 判别模型:
> CRF, LR, LogitsRegression,SVM,DNN

3. GMM-HMM中GMM为生成模型。根据语音帧序列的features构成的高斯分布来对状态进行建模。P(x|y)即为GMM的概率密度分布，通过求的GMM的各个分量的系数，均值，方差，即可完成GMM的模型构建。

4. DNN-HMM中DNN为判别模型，DNN输入是每一帧的40features，输出即表示该帧对应的logits，大小为决策树合并聚类的三状态的个数，经过softmax，可以求的最大概率对应的状态类别。



## kaldi中GMM和HMM迭代过程
- 单音速训练过程：
1. 按照输入features（T，D），对应音速三状态，进行帧数均分。（类似Kmeans初始化）。
2. 一个HMM状态可以多个帧。使用多个帧的features对该HMM状态中的观测概率分布进行建模，通用使用GMM。
   
    > GMM计算过程：(高斯混合模型)，混合系数 $\pi$ , 均值，协方差矩阵为参数。E-M算法更新。 
    > 1. 计算分模型k对观测数据yi的响应度r。E：step
    $$\gamma_{jk}=\frac{a_{k}\varPhi(y_{i}|\theta)}{{\Sigma}a_{k}\varPhi(y_{i}|\theta)}$$
    > 2. 根据相应度，迭代新一论参数。M：step
    $$ \mu_{k}=\frac{{\Sigma}\gamma_{jk}y_{j}}{{\Sigma}_{j=1}\gamma_{jk}} $$
    $$ \sigma^2=\frac{{\Sigma}\gamma_{jk}(y_{j}-\mu_{k})^2}{{\Sigma}_{j=1}\gamma_{jk}} $$
    $$ a_{k} = \frac{{\Sigma}\gamma_{jk}}{N} $$

3. 对HMM进行学习建模，数转移状态转移cout，得到发射矩阵，转移矩阵。即为HMM参数。
4. 重对齐: Vertibi算法进行解码，预测输入freatures实际的状态序列。
5. 由一个状态对应的多个帧继续对GMM状态进行更新。循环依次迭代。

  