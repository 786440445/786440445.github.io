---
title: tensorflow之滑动平均模型
catalog: true
date: 2019-05-10 15:47:34
subtitle: 滑动平均模型
header-img:
tags: tensorflow
---

# 滑动平均模型
tf.train.ExponentialMovingAverage(decay, step)
decay = min{decay, 1+step/(10+step)}
shadow_variable = decay * shadow_variable + (1-decay) * v1
ema.average(v1) = shadow_variable
shadow_variable为影子变量，variable为待更新的变量，decay为衰减率
decay决定了模型的更新速度，decay越大，模型越稳定

```python
# 滑动平均模型
import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)
# 模拟轮数，控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类class，初始化时给定了衰减率，和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时
# 列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 初始值为(0, 0)
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量的值 为5
    sess.run(tf.assign(v1, 5))

    # 衰减率 = min{0.99, (1+step)/(10+step)} = 0.1
    # 数值为0.1*0 + 0.9*5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))

    # 衰减率 = min{0.99, (1+step)/(10+step)} = 0.99
    # 数值为0.99*4.5 + 0.01*10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 衰减率 = min{0.99, (1+step)/(10+step)} = 0.99
    # 数值为0.99*4.555 + 0.01*10 = 4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
```
