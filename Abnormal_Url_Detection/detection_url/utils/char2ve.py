#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time  : 2019/5/23 10:35
# @FileName: char2ve.py

import logging
import numpy as np

from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Char2Vec(object):
    def __init__(self, size, window, min_count, workers, iter):
        """
        :param size: 向量的维数
        :param window: 句中当前词与预测词之间的最大距离
        :param min_count: 忽略总频率低于此值的所有单词
        :param workers: 线程
        :param iter: 语料库上的迭代次数
        """
        self._model = Word2Vec(
            size=size,
            window=window,
            min_count=min_count,
            workers=workers,
            iter=iter
        )

    def train(self, sentences):  # sentences是可迭代的列表str
        self._model.build_vocab(sentences=sentences)  # 从一系列的句子中建立词汇
        # 从一系列句子中更新模型的神经权重
        self._model.train(
            sentences=sentences,
            total_examples=self._model.corpus_count,  # 语料库统计
            epochs=self._model.epochs
        )

    def transform(self, char):
        #
        if char in self._model.wv.vocab:
            wv = self._model.wv[char]
        else:
            wv = np.random.normal(loc=1, scale=1, size=[self._model.vector_size])
        return wv

    def most_similar(self, char):
        return self._model.wv.most_similar(positive=char, topn=3)

    def load(self, path):
        self._model = Word2Vec.load(path)

    def save(self, path):
        self._model.save(path)

    def write_vocab(self, vocab_file, vocab_vector_file):
        vector = []
        with open(vocab_file, mode='w+', encoding='utf-8') as f:
            for k in self._model.wv.vocab:
                # 删除空格等操作符
                if k.strip() == '':
                    continue
                f.write(k + '\n')
                vector.append(self.transform(k))
        np.save(vocab_vector_file, np.array(vector))
