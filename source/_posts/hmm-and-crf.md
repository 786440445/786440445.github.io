---
title: hmm_and_crf
catalog: true
date: 2020-04-02 16:13:23
subtitle: 隐马尔可夫和条件随机场
header-img:
tags: 机器学习
---

# HMM分词

> 解决三个问题：模型λ=（A，B，π）。A为状态转移概率矩阵，B是观测概率矩阵。π是初始状态概率向量。
> π和A决定状态序列。B决定观测序列。O为观测序列，I为状态序列。

1. 概率计算问题：已知（A，B，π），和O观测序列。求：O观测序列出现概率P(O|λ)。 
2. 学习问题：已知O观测序列，估计模型的参数λ，使P(O|λ)最大。
3. 预测问题：也就是解码问题。已知λ=（A，B，π）和观测序列O，求给定观测序列条件概率P（I|O）最大的状态序列I。

- 篱笆网络（Lattice）求最短路径问题采用维特比算法进行解码。维特比算法是采用动态规划求最短路径的方案。实现的时间复杂度为O(d*N^2)。d为序列长度，N为状态数量。
- 根据一条路径A{1,2,3,4}-B{1,2,3,4}-C{1,2,3,4}-D{1,2,3,4}通过维特比算法求其最短路径方案。A{1-4}-B{1-4}有16种方案。保留其A{1-4}-B1最优的一种。保留A{1-4}-B2最优的一种，保留A{1-4}-B3最优的一种，保留A{1-4}-B4最优的一种。根据这四种继续往C计算16种，再保留分别经过C1，C2，C3，C4的最优的一种。直到D。这就是维特比算法。每次16中路径中保留了最优的四种方案。时间复杂度为O（序列长度乘以状态个数的平方）。

# CRF(Condition Random Field)

定义：条件随机场是给定随机变量X条件下，随机变量Y的马尔科夫随机场。满足 ![[公式]](https://www.zhihu.com/equation?tex=P%28Y_%7Bv%7D%7CX%2CY_%7Bw%7D%2Cw+%5Cneq+v%29+%3D+P%28Y_%7Bv%7D%7CX%2CY_%7Bw%7D%2Cw+%5Csim+v%29+) 的马尔科夫随机场叫做条件随机场（CRF）

# 概率图模型：

### 有向图模型：

![v2-5b3f6b4a2d905297b7f73a89e92ee618_r](\img\article\v2-5b3f6b4a2d905297b7f73a89e92ee618_r.jpg)

联合概率P（X）=p(x1)p(x2|x1)p(x3|x2)p(x4|x2)p(x5|x3,x4)

### 无向图模型：

![wuxiangtu](\img\article\wuxiangtu.png)

可以用因子分解将 ![[公式]](https://www.zhihu.com/equation?tex=P%3D%28Y%29) 写为若干个联合概率的乘积。咋分解呢，将一个图分为若干个“小团”，注意每个团必须是“最大团”。结果为所有最大团上势函数的乘积。

![](\img\article\wuxiangtugailv.png)

Z(x)为规范化因子，即![wuxiangtuguifanhuayinzi](\img\article\wuxiangtuguifanhuayinzi.png)

所以对于上图的无向图而言，

![wuxiangtugailv1](\img\article\wuxiangtugailv1.png)

其中![[公式]](https://www.zhihu.com/equation?tex=+%5Cpsi_%7Bc%7D%28Y_%7Bc%7D+%29) 是一个最大团 ![[公式]](https://www.zhihu.com/equation?tex=C) 上随机变量们的联合概率，一般取指数函数的：

![zuidatuan](\img\article\zuidatuan.png)

管这个东西叫做势函数。其中有点CRF的影子。

那么概率无向图的联合概率分布可以在因子分解下表示为：

![yinshifenjie](\img\article\yinshifenjie.png)
