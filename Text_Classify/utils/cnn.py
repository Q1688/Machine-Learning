#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Author : lianggq
# @Time  : 2019/7/17 14:20
# @FileName: cnn.py
import tensorflow as tf
from article_categories.config.config import FLAGS


class CNN(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, [None, FLAGS.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, FLAGS.num_classes], name="input_y")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        self.cnn()

    def cnn(self):
        # 词向量映射
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # cnn模型
        with tf.name_scope("cnn"):
            # 第一层卷积  conv1
            conv1 = tf.layers.conv1d(inputs=embedding_inputs,
                                     filters=FLAGS.conv1_num_filters,
                                     kernel_size=FLAGS.conv1_kernel_size,
                                     padding="SAME",
                                     name="conv1")

            # 第二层卷积  conv2
            conv2 = tf.layers.conv1d(inputs=conv1,
                                     filters=FLAGS.conv2_num_filters,
                                     kernel_size=FLAGS.conv2_kernel_size,
                                     padding="SAME",
                                     name="conv2")

            pool = tf.reduce_max(conv2, reduction_indices=[1], name="pooling")

            # 全连接层，后接dropout及relu激活
            full_c = tf.layers.dense(pool, units=FLAGS.hidden_dim, name="full_c")
            full_c = tf.contrib.layers.dropout(full_c, self.dropout_prob)
            full_c = tf.nn.relu(full_c, name="relu")

            # 输出层
            self.logits = tf.layers.dense(full_c, units=FLAGS.num_classes, name="logits")
            # 预测类别
            self.y_pred = tf.argmax(self.logits, axis=1)

        # 训练优化
        with tf.name_scope("training_op"):
            # 交叉熵和损失函数
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(xentropy)
            # 优化
            optimiaer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.optim = optimiaer.minimize(self.loss)

        # 计算准确率
        with tf.name_scope("accuracy"):
            correct = tf.equal(tf.argmax(self.input_y, axis=1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))
