#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time  : 2019/5/23 11:02
# @FileName: run_model.py
import os
import tensorflow as tf
from detection_url.config.config import FLAGS
from detection_url.run_model.Rnn_Net import Rnn_Net, train
from detection_url.utils.utils import get_iterator

if __name__ == '__main__':
    if not os.path.exists(FLAGS.word2vec_model_path):
        raise OSError('word2vec model file does not exist')

    iterator = get_iterator(
        buffer_size=FLAGS.batch_size,
        batch_size=FLAGS.batch_size,
        random_seed=666
    )

    # 输出iterator结果：BatchedInput(initializer=<tf.Operation 'MakeIterator' type=MakeIterator>, source=<tf.Tensor 'IteratorGetNext:0'
    # shape=(?, 200) dtype=int32>, target_input=<tf.Tensor 'IteratorGetNext:1' shape=(?, 1) dtype=int32>,
    # source_sequence_length=<tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=int32>, input_source_file=<tf.Tensor 'input_source_file:0'
    # shape=<unknown> dtype=string>, input_target_file=<tf.Tensor 'input_target_file:0' shape=<unknown> dtype=string>)

    net = Rnn_Net(
        embedding_size=FLAGS.embedding_size,
        iterator=iterator
    )

    with tf.Session() as sess:
        train(net, sess)
