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

## kaldi

- 模型初始化：
```shell
# 单音速模型训练
if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_2kshort data/lang_nosp exp/mono
fi
# boost-silence 1.25: 表示
```

1. train_mono.sh内部过程
- 初始化单音素模型。调用gmm-init-mono，生成0.mdl、tree。
- 调用compile-train-graph生成text中每句抄本对应的fst，存放在fsts.JOB.gz
- 第一次对齐数据。调用align-equal-stats-ali生成对齐状态序列，通过管道传递给gmm-acc-stats-ali，得到更新参数时用到的统计量
- 第一次更新模型参数。调用gmm-est更新模型参数。
- 进入训练模型的主循环：在指定的对齐轮数，使用gmm-align-compiled对齐特征数据，得到新的对齐状态序列；每一轮都调用gmm-acc-stats-ali计算更新模型参数所用到的统计量，然后调用gmm-est更新模型参数，并且在每一轮中增加GMM的分量个数。

##