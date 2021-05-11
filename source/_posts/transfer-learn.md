---
title: Inception-v3之迁移学习
toc_nav_num: true
catalog: true
date: 2019-05-20 23:50:02
subtitle: 迁移学习
header-img: 
tags: tensorflow
---

# Inception-v3模型
1. 将ImageNet上训练好的Inception-v3模型转移到另一个图像分类数据集上
1. 所谓迁移学习就是将一个问题上训练好的模型通过简单的调整使其适用于一个新的问题。
1. 本案例在Inception-v3保留所有卷积层参数，替换其最后一层全链接层

```python
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3瓶颈结点个数
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JEPG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = './inception_dec_2015/'
MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = './cache/bottleneck'
INPUT_DATA = './flower_photos'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


# 从数据文件夹中读取所有图片列表，并按照训练，验证，测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    # os.walk 得到目录下的所有子目录和文件
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []

        # 返回path最后的文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # 匹配所有的符合条件的文件，并将其以list的形式返回。
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()

        training_images = []
        testing_images = []
        validation_images = []

        # 随机构造一个训练集80%，测试集10%，验证集10%
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]

    category_list = label_lists[category]

    mod_index = index % len(category_list)

    base_name = category_list[mod_index]

    sub_dir = label_lists['dir']

    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 将当前图片作为输入计算瓶颈张量的值，这个瓶颈张量的值就是这个图的新的特征向量
def run_botteneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 将所有的图片处理成数值向量形式存入
def get_or_create_bottleneck(sess, image_lists, label_name, index,
                             category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_botteneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回特征向量
    return bottleneck_values


# 随机获取任意种类，指定category类型的数据量为how_many的数据
# 返回数据bottlenecks = [HowMany, BOTTLENECK_TENSOR_SIZE]
# ground_truths = [HowMany, n_classes]
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index,
                                              category, jpeg_data_tensor, bottleneck_tensor)

        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0

        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 获取全部测试数据
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())

    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                                                  jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    # 返回种类数量
    n_classes = len(image_lists.keys())

    # 读取已经训练好的Inception-v3模型、
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载读取的Inception-v3模型，返回数据输入所对应的张量以及计算瓶颈对应的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME, JEPG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(
        tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(
        tf.float32, [None, n_classes],
        name='GroundTruthInput')

    # 最后一层全链接解决新的图片分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        # 训练过程
        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH,
                'training', jpeg_data_tensor, bottleneck_tensor)

            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks,
                                            ground_truth_input: train_ground_truth})

            # 在验证数据上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottleneck, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH,
                    'validation', jpeg_data_tensor, bottleneck_tensor)

                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottleneck,
                    ground_truth_input: validation_ground_truth})

                print('Step %d : Validation accuracy on random sampled %d examples = %.1f%%' % (i, BATCH, validation_accuracy * 100))

        # 在测试数据上测试正确率
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)

        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                             ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()

```