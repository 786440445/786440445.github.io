---
title: C-FSMN(简洁的FSMN)
catalog: true
date: 2019-07-10 20:39:30
subtitle: 简介的FSMN
header-img:
tags: 语音识别
---

# C-FSMN
## 模型结构
![](/img/article/c-fsmn_1.png)

## 模型介绍
1. FSMN相比于FNN，需要将记忆模块的输出作为下一个隐层的额外输入，这样就会引入额外的模型参数。隐层包含的节点越多，则引入的参数越多。
1. 我们通过结合矩阵低秩分解（Low-rank matrix factorization）的思路，提出了一种改进的FSMN结构，称之为简洁的FSMN（Compact FSMN，cFSMN）
1. 对于C-FSMN，通过在网络的隐层后添加一个低维度的线性投影层，并且将记忆模块添加在这些线性投影层上。即h(l, t)到p(l, t)经过一次线性变换。
1. 进一步的，cFSMN对记忆模块的编码公式进行了一些改变，通过将当前时刻的输出显式的添加到记忆模块的表达中，从而只需要将记忆模块的表达作为下一层的输入。这样可以有效的减少模型的参数量，加快网络的训练。
1. 下一层的结果h(l+1,t) = U’*P’(t)

具体的，单向和双向的cFSMN记忆模块的公式表达分别如下：
![](/img/article/c-fsmn_2.png)
