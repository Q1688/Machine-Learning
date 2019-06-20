#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time  : 2019/5/22 18:19
# @FileName: config.py
import tensorflow as tf
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

FLAGS = tf.app.flags.FLAGS
"""
about URL clear config
"""
# 保存URL字符串的文件
tf.app.flags.DEFINE_string("train_src_file", "../data/source.txt", "train source file dir")
# 保存URL标签的文件
tf.app.flags.DEFINE_string("train_tgt_file", "../data/target.txt", "train target file dir")
# URL根据char2ve处理的配置文件
tf.app.flags.DEFINE_string("word2vec_model_path", "../data/char2ve/vector.bin", "word to vector model path")
tf.app.flags.DEFINE_string("vocab_file", "../data/char2ve/vocab.txt", "vocab file dir")
tf.app.flags.DEFINE_string("vocab_vector_file", "../data/char2ve/vocab_vector.npy", "vocab vector file")

tf.app.flags.DEFINE_string("model_path", "../model_filt/", "model path")
tf.app.flags.DEFINE_string("test_src_file", "../data/source_test.txt", "test source file dir")
tf.app.flags.DEFINE_string("test_tgt_file", "../data/target_test.txt", "test target file dir")

tf.app.flags.DEFINE_string("model_pb_file", "../data/abnormal_detection_model.pb", "converted model file")

"""
公共配置
"""
tf.app.flags.DEFINE_integer("embedding_size", 32, "vocab vector embedding size")
# 数据集的连续元素的个数,并组合成一个单批.
tf.app.flags.DEFINE_integer("batch_size", 2, "batch size")
tf.app.flags.DEFINE_integer("num_steps", 100, "number of input string max length")
# 训练的次数
tf.app.flags.DEFINE_integer("epoch", 100, "number of training epoch")

"""
RNN层配置
"""
tf.app.flags.DEFINE_integer("num_layers", 3, "number of rnn layer")
tf.app.flags.DEFINE_integer("num_hidden", 15, "hidden layer output dimension")
tf.app.flags.DEFINE_float("input_keep_prob", 0.5, "input keep prob")
tf.app.flags.DEFINE_float("output_keep_prob", 0.5, "output keep prob")
tf.app.flags.DEFINE_float("state_keep_prob", 1.0, "state keep prob")

tf.app.flags.DEFINE_string("tb_path", "./tb/", "tensorboard file path")

"""
学习速率配置
"""
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("decay_steps", 5, "decay steps")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "decay rate")

"""
cpu core config
"""
tf.app.flags.DEFINE_integer("cpu_num", 4, "cpu core number")

