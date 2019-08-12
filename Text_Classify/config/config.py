#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Author : lianggq
# @Time  : 2019/7/15 15:30
# @FileName: config.py
import tensorflow as tf

# 添加配置文件，设置配置文件时一定要注意（DEFINE_integer/string....）类型
FLAGS = tf.app.flags.FLAGS

# 类别数
tf.app.flags.DEFINE_integer("num_classes", 58, "number of labels")
# 词汇表大小
tf.app.flags.DEFINE_integer("vocab_size", 100000, "vocab size")
# 序列长度
tf.app.flags.DEFINE_integer("seq_length", 200, "max length of sentence")
# drop保留比例
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.9, "Drop retention ratio")
# 学习率
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
# 每批训练大小
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
# 总迭代轮次
tf.app.flags.DEFINE_integer("num_epochs", 10, "Total number of iterations")
# 每多少轮输出一次结果vocab_file
tf.app.flags.DEFINE_integer("print_per_batch", 100, "Output results every number of rounds")
# 每多少轮存入tensorboard
tf.app.flags.DEFINE_integer("save_per_batch", 5, "Every number of rounds to the tensorboard")

"""CNN算法模型配置"""
# 词向量维度
tf.app.flags.DEFINE_integer("embedding_dim", 200, "word vector dimension")
# 第一个卷积层参数
tf.app.flags.DEFINE_integer("conv1_num_filters", 256, "The first convolution layer kernel number")
tf.app.flags.DEFINE_integer("conv1_kernel_size", 4, "The first convolution layer convolution kernel size")
# 第二个卷积层参数
tf.app.flags.DEFINE_integer("conv2_num_filters", 128, "The second convolution layer kernel number")
tf.app.flags.DEFINE_integer("conv2_kernel_size", 2, "The second convolution layer convolution kernel size")
# 全连接层神经元
tf.app.flags.DEFINE_integer("hidden_dim", 128, "Full connectome neurons")

"""数据处理文件配置"""
# 训练数据
tf.app.flags.DEFINE_string("train_data", "../data/train.txt", "train data")
# 测试数据
tf.app.flags.DEFINE_string("test_data", "../data/test.txt", "test data")
# 验证数据
tf.app.flags.DEFINE_string("val_data", "../data/val.txt", "val data")
# 分离出的训练标签数据
tf.app.flags.DEFINE_string("train_label_file", "../data/train_clean_label.txt", "train data label")
# 分离出的训练数据
tf.app.flags.DEFINE_string("train_content_file", "../data/train_clean_content.txt", "clear train data")
# 分离出的测试标签数据
tf.app.flags.DEFINE_string("test_label_file", "../data/test_clean_label.txt", "test data label")
# 分离出的测试数据
tf.app.flags.DEFINE_string("test_content_file", "../data/test_clean_content.txt", "clear test data")
# 分离出的验证标签数据
tf.app.flags.DEFINE_string("val_label_file", "../data/val_clean_label.txt", "val data label")
# 分离出的验证数据
tf.app.flags.DEFINE_string("val_content_file", "../data/val_clean_content.txt", "clear val data")

# 清洗后的高频前10000词汇表
tf.app.flags.DEFINE_string("vocab_filename", "../data/word2vc/vocab.txt", "vocabulary")
# 词概率(Word2vec模型)
tf.app.flags.DEFINE_string("vector_word_filename", "../data/word2vc/vector_word.txt", "vector_word trained by word2vec")
# 词向量表
tf.app.flags.DEFINE_string("vector_word_npz", "../data/word2vc/vector_word_npz", "save vector_word to numpy file")
