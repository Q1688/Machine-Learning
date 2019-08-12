#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Author : lianggq
# @Time  : 2019/7/19 10:17
# @FileName: word2vec.py

from article_categories.config.config import FLAGS
from article_categories.utils.tools import read_data, write_data_clean, build_vocab
from gensim.models import word2vec
import logging
import os

if __name__ == '__main__':
    # 训练数据的处理
    train_data = FLAGS.train_data
    label, content = read_data(train_data)
    # 对训练数据进行分词处理
    write_data_clean(label, content, FLAGS.train_label_file, FLAGS.train_content_file)

    # 训练词向量训练词向量,此处更该了split()方法，默认为空格，修改为逗号
    sentences = word2vec.LineSentence(FLAGS.train_content_file)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=50, window=5, min_count=1, workers=4, iter=4)
    model.wv.save_word2vec_format(FLAGS.vector_word_filename, binary=False)

    if not os.path.exists(FLAGS.vocab_filename):
        # 构建词汇表和词汇长度表
        build_vocab(FLAGS.train_content_file, FLAGS.vocab_filename, FLAGS.vector_word_npz)

    # 测试数据的处理
    test_data = FLAGS.test_data
    label, content = read_data(test_data)
    # 对测试数据进行分词处理
    write_data_clean(label, content, FLAGS.test_label_file, FLAGS.test_content_file)

    # 验证数据的处理
    val_data = FLAGS.val_data
    label, content = read_data(val_data)
    # 对验证数据进行分词处理
    write_data_clean(label, content, FLAGS.val_label_file, FLAGS.val_content_file)
