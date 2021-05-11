---
title: tacotron2
catalog: true
date: 2021-02-10 19:29:53
subtitle: Tacotron2
header-img: 
tags: 语音合成
---

## Tacotron1：
### input_embedding

### prenet

- prenet是通过 使用两层全连接映射 实现 [N,T,256] -> [N,T,256] -> [N,T,128]作用是对输入进行初步特征提取和dropout以提升泛化能力

```python
def prenet(inputs, is_training, layer_sizes, scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
  return x
```

### encoder_cbhg
- encoder_cbhg是输入端的cbhg结构: 包含三部分 
1. Convolution Bank
- 这一部分对于特征提取的思路是用height=1, width=word_dim的filter 过滤得到每一个input_embedding的特征值，然后相邻两个word一起过滤得到局部窗口为2的上下文特征值，即2-gram，一直提取出 16-gram。128个filter用于关注不同的特征值。
- 大体上和textCNN的思路差不多，都是n-gram的思想，不同的是这里面进行卷积操作的时候使用的padding=‘same’ 为了保证对齐，textCNN里面我们用的时候使用 padding='valid' 然后用reduce_max进行对齐。
- encoder_cbhg使用的 kernel_size 为 [1,2,3,...,16] filter_size = 128
每个conv1d 结果按照 axis = -1 拼接起来后的shape 为 [batch_size, Time_stmp,128*16]
2. Highway Network
- 在residual connection 后 接了4层 highway net。highway net 就是加速的深层DNN。result shape = [batch_size,Time_stmp,128]
3. BiGRU
- 输入是 highway net 的输出 shape = [batch_size,Time_stmp,128] 经过双向rnn 后 由于 num_units = 128 , 双向拼接后为 256 故而 result shape = [batch_size,Time_stmp,256]


# Tacotron2

## Encoder

## Decoder

## Attention
 