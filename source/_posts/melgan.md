---
title: melgan
catalog: true
toc_nav_num: true
mathjax: true
date: 2021-05-01 01:57:18
subtitle: MelGan
header-img:
tags: TTS
---

# MelGan
通过输入mel谱特征，生成wav数据，使用真实wav数据进行判别训练。
loss1：hingloss衡量生成wav与实际wav的差异。
loss2：衡量真实


## Generator
- 多层Conv1d: 对Mel频谱时域信息进行建模。使用weight norm能够让learning rate自适应/自稳定。
- 2 * (Unsampling Layer + Residal Block):
1. Unsampleing Layer：
2. Residual Block：
   1. 

- 由对齐的mel谱进行上采样到音频，采用反一维卷积完成，帧数*上采样倍数=采样点个数

## Descrimtor 
