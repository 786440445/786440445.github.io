---
title: chain_model
catalog: true
comments: true
indexing: true
header-img: ../../../../img/default.jpg
top: false
tocnum: true
date: 2021-05-26 17:48:29
subtitle:
tags:
categories:
---

#  chain model训练过程

> 单音素训练
- train_mono.sh
    ```shell
    steps/train_mono.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/mono
    # params：
    1. train_data
    2. lang
    3. exp模型目录
    ```

> 构图
- mkgraph.sh
    ```shell
    utils/mkgraph.sh data/lang_test exp/mono exp/mono graph
    # params：
    1. lang_test
    2. exp模型目录
    3. exp构图目录
    ```

> 解码
- decode.sh
    ```shell
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/mono/graph data/test exp/mono/decode_test
    ```

> 对齐，生成ali文件
- align_si.sh
    ```shell
    steps/align_si.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/mono exp/mono_ali
    # params：
    1. train_data
    2. lang
    3. exp模型目录
    4. exp对齐结果
    ```

> 三音素训练
- train_deltas.sh
    ```aishell
    steps/train_deltas.sh --cmd "$train_cmd" 2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;
    # params：
    1. GMM个数
    2. 决策树状态建模结点个数
    3. train
    4. lang
    5. exp对齐结果
    6. exp模型
    ```
- mkgraph.sh
- decode.sh
- align_si.sh

> 三音素训练
- train_deltas.sh
- mkgraph.sh
- decode.sh
- align_si.sh

> do the alignment with fMLLR.
- train_lda_mllt
- mkgraph.sh
- decode.sh
- align_fmllr.sh

> Building a larger SAT system.
- train_sat.sh
- mkgraph.sh
- decode_fmllr.sh
- align_fmllr.sh
- local/chain/run_tdnn.sh

# run_tdnn.sh

- local/nnet3/run_ivector_common.sh

- align_fmllr_lats.sh
    ```shell
    steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set data/lang exp/tri5a exp/tri5a_sp_lats
    ```

- gen_topo.py
    ```shell
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
    # params：
    1. nonsilphonelist: $(cat $lang/phones/silence.csl) 
    2. silphonelist: $(cat $lang/phones/nonsilence.csl) 
    ```
- build_tree.sh
    ```shell
    steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
    # params：
    1. frame-subsampling-factor 下采样帧率
    2. context-opts：
    3. data
    4. lang
    5. ali_dir
    6. treedir
    ```

> creating neural net configs using the xconfig parser";

    the first splicing is moved before the lda layer, so no splicing here
    relu-batchnorm-layer name=tdnn1 dim=625
    relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
    relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
    relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
    relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
    relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625

> train.py
    
    steps/nnet3/chain/train.py