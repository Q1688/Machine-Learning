#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time  : 2019/5/22 19:31
# @FileName: run_url_mian.py
import os

import pandas as pd
from detection_url.tools.loger import logger
from detection_url.utils.char2ve import Char2Vec
from detection_url.utils.utils import parse_http_request, write_src_tgt
from detection_url.config.config import FLAGS

if __name__ == '__main__':
    try:
        primary_df = pd.read_csv('../data/primary.csv')
    except Exception as reason:
        logger.error(reason)

    # 将测试数据中URL进行列表化
    source_list = primary_df['request_url'].values.tolist()
    # 将训练数据中URL的标签进行列表化
    target_list = primary_df['status'].values.tolist()
    # url_parse_list = parse_http_request(source_list)
    # 将URL转化成字符串，并将训练中URL和标签保存到不同的文件中
    url_char_list = write_src_tgt(source_list, target_list,
                                  FLAGS.train_src_file, FLAGS.train_tgt_file)
    # 将URL根据char2vec进行向量处理
    model = Char2Vec(size=FLAGS.embedding_size, window=5, min_count=1, workers=2, iter=5)
    model_path = FLAGS.word2vec_model_path
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        model.train(sentences=source_list)
        model.save(model_path)
    model.write_vocab(FLAGS.vocab_file, FLAGS.vocab_vector_file)
