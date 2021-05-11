---
title: TextClassify
catalog: true
date: 2019-12-05 12:32:13
subtitle: 文本分类，关键词提取
header-img:
tags: 机器学习
---

# TextRank算法

## pagerank

- TextRank来源于PageRank算法，PageRank算法如下所示：

<img src="\img\article\pagerank01.png" style="zoom: 67%;" />

1. d为固定系数0.85，S(Vi)表示Vi的权重，In[Vi]为Vi的入度的节点集合，Out(Vj)为Vj出度的节点的集合。绝对值表示其个数。
2. 迭代n次后，网页中节点的权重参数不再变化，最后每一个节点的S(Vi)即为该节点的重要性。

## TextRank

- 若在pagerank中将每个图中的节点换成句子，或者是关键词，则可以演进为TextRank算法。（计算公式如下所示d=0.85

  ![image-20191120144110314](\img\article\image-20191120144110314.png)

-  对于摘要提取过程，图的信息主要来自于边的权重，边的权重是句子i和句子j的相似度。所以在迭代的过程中需要乘上边的权重。 

- 两个句子的相似程度计算如下：

  <img src="\img\article\textrank-01.png" style="zoom: 67%;" />

  1.  分子是在两个句子中都出现的单词的数量。|Si|是句子i的单词数。 
  2. 这里采用jieba分词对每两个句子之前进行计算相似度。

- 整体过程就是一个迭代过程，一次遍历，就是遍历所有节点。


## 关于textrank算法在弹幕提取中的应用：

1. 首先根据每一个直播间的前8000条弹幕（按频数选取，从高到底选取8000条，不用全部，全部都有几千万乃至上亿条，数据量过大不需要）。
2. 初始设置特殊弹幕的权重为1，其他所有弹幕权重为0.5。
3. 计算每两个句子之间的相似度，这里按照上式的公式来计算。
4. 按照textrank算法迭代，直到图收敛。
   - 一次遍历计算所有句子的得分：根据(1-d)+d*所有入度的句子里的
5. 最后选出top50的弹幕作为自己提取的关键性弹幕。

# TF-IDF算法

1. Text Frequency：统计出现次数最多的词：关键句在该文档中出现的频数

2. Inverse Document Frequency：IDF逆文档频率，log(文档的总数量/关键句在文档中出现次数+1)

## 关于tf-idf在弹幕提取中的应用

1. 判断当前文本在当前直播间所有弹幕中出现的频数。这里需要根据文本相似度进行计算。不能直接根据弹幕的频数计算，因为，从数据库中读取的数据都是合并后的topN数据。
2. 计算逆文档频率。log(sum/count+1)。sum表示文档总数，count表示当前弹幕在其他文档里出现的次数（加1是为了防止一次都没有出现）
3. TF*IDF就表示最后的弹幕的权重。按照权重topN来进行提取选择。

```python
from collections import Counter
import math

class TfIdf:
    def __int__(self):
        pass

    def calc_tf(self, lst):
        if_dict = Counter(lst)
        return if_dict

    def calc_idf(self, lst, document_lst):
        num = len(document_lst)
        count = 0
        idf_dict = {}
        for item in lst:
            for doc in document_lst:
                if doc.get(item):
                    count += 1
            idf_dict[item] = math.log(num/(count+1))
        return idf_dict

    def get_tf_idf_scores(self, tf_scores, idf_scores, top_num):
        result = {}
        for k, (content, tf) in enumerate(tf_scores.items()):
            idf = idf_scores.get(content)
            scores = tf * idf
            result[content] = scores
        result = sorted(result.items(), key=lambda x:x[1], reverse=True)
        result = result[:top_num]
        return result
```

# LDA算法

## （Latent Dirichlet Allocation）:潜在狄利克雷分布

- 它是一种无监督的贝叶斯模型。文档生成模型

- 认为一偏文档是有多个主题的，每个主题又对应着不同的词（这里可以理解为关键句）。一篇文档的构造过程，首先是一定的概率选择某个主题。 然后再在这个主题下以一定的概率选出某一个词，这样就生成了这篇文档的第一个词。不断重复这个过程，就生成了整篇文章 。（文档——主题——词）
