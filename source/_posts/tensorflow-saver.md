---
title: tensorflow模型持久化
toc_nav_num: true
catalog: true
date: 2019-05-13 23:03:42
subtitle: 持久化
header-img:
tags: tensorflow
---

# 保存模型
1. saver.save可以将tensorflow模型保存到路径下
1. tensorflow会将计算图的结构和图上参数取值分开保存
1. model.ckpt.meta保存了计算图的结构
1. model.ckpt保存了程序中每一个变量的取值
1. checkpoint保存了一个目录下所有的模型文件列表

## 保存模型code
```python
import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "./model/model.ckpt")
```
![](/img/article/saver_test1)

## 加载模型数据
```python
import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./model/model.ckpt")
    print(sess.run(result))

```

## 加载全部变量
```python
import tensorflow as tf

saver = tf.train.import_meta_graph("./model/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
```

# 保存加载部分变量
```python
import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
saver = tf.train.Saver({"v1": v1, "v2": v2})
```

# 整个计算图放在一个文件中
```python
import tensorflow as tf
from tensorflow.python.framework import graph_util
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init_op)
	graph_def = tf.get_default_graph().as_graph_def()
	output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

	with tf.gfile.GFile("./model/combined_model.pb", "wb") as f:
		f.write(output_graph_def.SerializeToString())
```

```python
import tensorflow as tf
from tensorflow.python.platform import  gfile

with tf.Session() as sess:
    model_filename = './model/combined_model.pb'
    # 读取保存的模型文件，解析成对应的protobuffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))

```
