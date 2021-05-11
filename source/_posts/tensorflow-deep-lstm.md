---
title: tensorflow实现深度循环LSTM
toc_nav_num: true
catalog: true
date: 2019-05-22 22:25:01
subtitle: 深度LSTM
header-img:
tags: tensorflow
---

# LSTM网络结构
## 单个lstm结构
![](/img/article/lstm_cell.png)
1. Cell，就是我们的小本子，有个叫做state的参数东西来记事儿的
1. Input Gate，Output Gate，在参数输入输出的时候起点作用，算一算东西
1. Forget Gate：不是要记东西吗，咋还要Forget呢。这个没找到为啥就要加入这样一个东西，因为原始的LSTM在这个位置就是一个值1，是连接到下一时间的那个参数，估计是以前的事情记太牢了，最近的就不住就不好了，所以要选择性遗忘一些东西。（没找到解释设置这个东西的动机，还望指正）

## BasicLSTMCell
将forget_bias（默认值：1）添加到忘记门的偏差(biases)中以便在训练开始时减少以往的比例(scale)。该神经元不允许单元裁剪(cell clipping),投影层，也不使用peep-hole连接，它是一个基本的LSTM神经元。想要更高级的模型可以使用：tf.nn.rnn_cell.LSTMCell。

## embedding_lookup

## 整个过程原理
1. 输入数据为[batch_size, num_steps],每个句子的单词个数为num_steps，由于句子长度就是时间长度，因此用num_steps代表句子长度。
1. 在NLP问题中，我们用词向量表示一个单词，这里VOCAB_SIZE表示整个向量长度，语料库中单词的个数是vocab_size
1. LSTM结构中是一个神经网络，这个网络的隐藏单元个数我们设为hidden_size，那么这个LSTM单元里就有4*hidden_size个隐藏单元，也就是隐含变量的维度。
1. lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)创建一个简单的LSTM
1. cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS) 创建多层LSTM
1. 




```python
import numpy as np
import tensorflow as tf

from tensorflow.models.tutorials.rnn.ptb import reader

DATA_PATH = './simple-examples/data'
# 隐藏层层数
HIDDEN_SIZE = 200
# LSTM结构层数
NUM_LAYERS = 2
# 词典规模
VOCAB_SIZE = 10000
# 学习率
LEARNING_RATE = 1.0
# batch大小
TRAIN_BATCH_SIZE = 20
# 训练数据截断长度
TRAIN_NUM_STEP = 35

# 在测试时不需要截断，所以可以将测试数据看成一个超长序列
# 测试数据batch大小
EVAL_BATCH_SIZE = 1
# 测试数据截断长度
EVAL_NUM_STEP = 1
# 使用训练数据轮数
NUM_EPOCH = 2
# 结点不被dropout的概率
KEEP_PROB = 0.5
# 控制梯度膨胀的参数
MAX_GRAD_NORM = 5


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        # 输入数据占位符,shape=[batch_size, 25]
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        # 目标输出占位符,shape=[batch_size, 25]
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        # 一个简单的LSTMcell，输入shape=[HIDDEN_SIZE, 1]
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        # 训练中设置dropout
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化最初状态
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        # 将单词ID转换为单词向量,总共有VOCAB_SIZE个单词，每个单词的维度为HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本batch_size * num_steps个单词ID转化为单词向量
        # 转化后的输入层维度为batch_size * num_steps * HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只有在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)
        outputs = []

        state = self.input_data
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        # 把输出队列展开成[batch, hidden_size * num_steps]
        # 再reshape成[batch * numsteps, hidden_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数
        loss = tf.contrib.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],   # 期待的答案，将[batch_size, num_steps]压缩为一维数组
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        self.cost = tf.reduce_mean(loss)/batch_size
        self.final_state = state

        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，避免梯度膨胀问题
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # 定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


def run_epoch(sess, model, data, train_op, output_log):
    # 计算perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    for step in range(model.num_steps):
        x, y = reader.ptb_producer(data, model.batch_size, step)
        cost, state, _ = sess.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y, model.initial_state: state})

        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs/iters)))
    return np.exp(total_costs/iters)


def main(_):
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的网络
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试用的网络
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i+1))
            run_epoch(sess, train_model, train_data, train_model.train_op, True)
            valid_perplexity = run_epoch(sess, eval_model, valid_data, tf.no_op(), False)
            print("Epoch : %d Validation Perplexity: %.3f" % (i+1, valid_perplexity))

        test_perplexity = run_epoch(sess, eval_model, test_data, tf.no_op(), False)
        print("Testing Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
```