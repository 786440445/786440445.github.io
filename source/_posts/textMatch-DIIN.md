---
title: 密集交互推断网络DIIN
catalog: true
date: 2020-04-09 11:19:33
subtitle: 深度学习
header-img:
tags: 文本匹配
---

## DIIN：密集交互推断网络

### 1. Embedding Layer:

将词向量，字符特征向量和句法特征向量 进行串联。

词向量可以是通过预训练模型word2vec拿到。

字符特征向量是将字符向量 经过一维卷积，然后最大池化得到。

句法特征向量由词性Pos特征向量和二进制精确匹配EM特征向量组成。如果在另一个句子中有与该词具有相同词干或相同词元的词，那么EM值会被激活。

### 2. Eocoding Layer:

这一层主要是对输入v进行highway network + self attention + fuse gate过程

1. attention的计算为W[a;a;axa]
2. 对attnetion进行softmax按照权重乘以v结果为attention
3. 将v与attention进行concat串联为v_hat。并计算z,r,f。z,r,f分别为v_hat的线性变换。
4. 其中z使用tanh激活函数。r,f使用sigmoid激活函数。
5. res = r\*v+f\*z

### 3. Interaction Layer

对p和h进行点乘，结果为I

### 4. Feature Extraction layer

将相关性结果I通过DenseNet进行特征提取

- 相比ResNet，DenseNet提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入

1. DenseNet的网络结构主要由DenseBlock和Transition组成
2. 在DenseBlock中，各个层的特征图大小一致，可以在channel维度上连接。DenseBlock中的非线性组合函数 ![[公式]](https://www.zhihu.com/equation?tex=H%28%5Ccdot%29) 采用的是**BN+ReLU+3x3 Conv**的结构
3. 对于Transition层，它主要是连接两个相邻的DenseBlock，并且降低特征图大小。Transition层包括一个1x1的卷积和2x2的AvgPooling，结构为**BN+ReLU+1x1 Conv+2x2 AvgPooling**

- 由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练。由于每层可以直达最后的误差信号，实现了隐式的[“deep supervision”](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.5185)；
- 参数更小且计算更高效，这有点违反直觉，由于DenseNet是通过concat特征来实现短路连接，实现了特征重用，并且采用较小的growth rate，每个层所独有的特征图是比较小的；
- 由于特征复用，最后的分类器使用了低级特征。

### 5. Output Layer

dense+softmax到分类类别个数上。对于匹配模型，就是0，1。