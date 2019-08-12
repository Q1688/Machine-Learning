#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Author : lianggq
# @Time  : 2019/7/24 16:33
# @FileName: prediction.py
import os

import jieba
jieba.load_userdict('../stopword/myjieba.txt')

from article_categories.config.config import FLAGS
from article_categories.utils.cnn import CNN
import tensorflow as tf
import tensorflow.contrib.keras as kr

from article_categories.utils.tools import bulid_vocab_id, build_class_id, build_stop_word
# 模型的路径
save_dir = "./model/"
save_path = os.path.join(save_dir, "best_validation")  # 最佳验证结果保存路径


class cnn_model(object):
    def __init__(self):
        self.label_list = [label for label in build_class_id(FLAGS.train_label_file).keys()]
        self.vocab, self.vocab_id = bulid_vocab_id(FLAGS.vocab_filename)
        self.model = CNN()

    def prediction(self, message):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph("./model/best_validation.meta")
            saver.restore(sess, save_path=save_path)

            data = [self.vocab_id[word] for word in message if word in self.vocab_id]

            feed_dict = {
                self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], FLAGS.seq_length),
                self.model.dropout_prob: 1.0
            }

            y_pred = sess.run(self.model.y_pred, feed_dict=feed_dict)
            return self.label_list[y_pred[0]]


if __name__ == '__main__':
    model = cnn_model()
    word_list = []
    # 引用停用分词
    stop_list = build_stop_word('../stopword/stopword.txt')
    # 获取预分类的数据
    with open('../prediction.txt', 'r', encoding='utf-8') as r:
        for line in r.readlines():
            content = line.strip()
            document_cut = jieba.cut(content, cut_all=False)
            for word in document_cut:
                if word not in stop_list:
                    word_list.append(word)
            print(model.prediction(word_list))
