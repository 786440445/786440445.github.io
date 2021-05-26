---
title: interview_0430
catalog: true
toc_nav_num: true
mathjax: true
date: 2021-05-01 01:57:55
subtitle:
header-img:
tags:
---

# 4.30 面试记录

## ASR:

- CE和CTC loss区别
  
CE是基于帧粒度对两个序列的概率分布的相似程度进行建模，粒度比较细。
$$
loss = -\frac{\Sigma_{i=0}{p(i)log(q(i))}}{N}
$$

CTC loss
引入blank符号，对进行序列的映射，从整体序列的维度进行考虑，但是CTC具有条件独立性，即前后帧条件独立，相互不影响。

RNN-A loss
类似CTC+RNN。one-one。在翻译的过程中，输入一帧，输出一帧，前后帧有时许关系，前帧可以影响后帧。

RNN-T loss
类似CTC+RNN。one-many。在翻译的过程中，输入一帧，输入多帧，直到遇到blank，选择下一个x，前帧可以影响后帧。


- 维特比算法

使用动态规划的思想，将大问题分解为字问题，即在第t时刻是最优的路径，则t-1时刻，也必然是最优路径。

在前一步的没个节点，选择出经过每个节点的D个状态的路径中的最优路径，作为下一时刻的前置子问题方案。下一个问题时，继续选择经过当前D个节点的D中的最优路径。

时间复杂度为O（L*D^2）,L为序列长度，D为可选择节点个数。

- token passing
在WFST解码过程中，使用token passing算法，分别对状态节点进行记忆和回溯，类似维特比算法。

- 1x1卷积和FC区别
1x1卷积就等于Fully connection


- 除了fbank特征，mfcc还有什么特征。

Linear Predict C：线性预测系数(LPC)

i-vector

x-vector。

## TTS

- MelGan的损失函数


- 什么是回归，自回归，分类与回归的区别

