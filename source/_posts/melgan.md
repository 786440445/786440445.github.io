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
## Generator
- 由对齐的mel谱进行上采样到音频，采用反一维卷积完成，帧数*上采样倍数=采样点个数

## Descrimtor 
