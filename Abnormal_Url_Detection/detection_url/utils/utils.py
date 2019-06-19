#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time  : 2019/5/22 18:20
# @FileName: utils.py
from urllib import parse
import tensorflow as tf
from detection_url.config.config import FLAGS
import numpy as np
import collections


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "source_sequence_length",
                                           "input_source_file",
                                           "input_target_file"))):
    pass


def parse_http_request(http_req_list):
    """
    对URL进行解析
    :param http_req_list: URL列表
    :return: 字符串列表
    """
    parsed_req_list = []
    for http_req in http_req_list:
        decoded = parse.unquote(http_req)
        parsed = parse.urlparse(decoded)
        if parsed.path == '':
            continue
        if parsed.query == '':
            parsed_req_list.append(parsed.path)
        else:
            parsed_req_list.append(parsed.path + '?' + parsed.query)
    return parsed_req_list


def write_src_tgt(src_list, tgt_list, src_file, tgt_file):
    """
    将URL解析成字符列表
    :param src_list: URL解析的列表
    :param tgt_list: URL对应的标签列表
    :param src_file: 保存URL字符列表的文件
    :param tgt_file: 保存URL字符列表标签的文件
    :return: 字符串列表
    """
    char_list = [[j for j in i] for i in src_list]

    with open(src_file, mode='w+', encoding='utf-8') as f:
        for i in char_list:
            f.write(' '.join(i) + '\n')
        src_lines = len(f.readlines())

    with open(tgt_file, mode='w+', encoding='utf-8') as f:
        for i in tgt_list:
            f.write(str(i) + '\n')
        tgt_lines = len(f.readlines())

    if src_lines != tgt_lines:
        raise ValueError('line number was different with write source and target in files')
    return char_list


def create_vocab_tables(vocab_file):
    """
    构建词汇文件
    :param vocab_file:创建的表名
    :return: 创建的表
    """
    vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file, num_oov_buckets=2)
    return vocab_table


def get_vocab_size(vocab_file):
    """
    统计词汇长度
    :param vocab_file: 词汇文件名
    :return: 长度
    """
    with open(vocab_file, encoding='utf-8') as f:
        count = len(f.readlines())
    return count


def get_iterator(batch_size, buffer_size=None, random_seed=None, num_threads=4, src_max_len=FLAGS.num_steps):
    input_source_file = tf.placeholder(dtype=tf.string, shape=None, name="input_source_file")
    input_target_file = tf.placeholder(dtype=tf.string, shape=None, name="input_target_file")
    # 词汇表映射
    vocab_table = create_vocab_tables(FLAGS.vocab_file)
    # 统计词汇表的长度vocab.txt
    vocab_size = get_vocab_size(FLAGS.vocab_file)

    if buffer_size is None:
        buffer_size = batch_size * 5

    # 生成一个src_dataset，src_dataset中的每一个元素就对应了文件中的一行
    src_dataset = tf.data.TextLineDataset(input_source_file)
    tgt_dataset = tf.data.TextLineDataset(input_target_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size, random_seed)

    # 分割数据
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values,
            tf.string_to_number(tf.string_split([tgt]).values, out_type=tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # 获取最大长度的数据
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # 词汇表查询
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt, tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # 计算文本行实际长度
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in: (
            src, tgt_in, tf.size(src)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([src_max_len]),
                           tf.TensorShape([1]),
                           tf.TensorShape([])),
            padding_values=(vocab_size + 1,
                            0,
                            0))

    batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_ids, src_seq_len) = (batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_ids,
        source_sequence_length=src_seq_len,
        input_source_file=input_source_file,
        input_target_file=input_target_file)


def get_num_classes(file):
    """
    标签种类
    :param file: 标签文件
    :return: 类型的总数
    """
    with open(file, encoding='utf-8') as f:
        count = len(set([i.strip('\n') for i in f.readlines()]))
    return count


def load_word2vec_embedding(vocab_size, embeddings_size):
    """
    加载外接的词向量。
    return:
    """
    embeddings = np.random.uniform(-1, 1, (vocab_size + 2, embeddings_size))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(666)
    unknown = np.asarray(rng.normal(size=embeddings_size))
    padding = np.asarray(rng.normal(size=embeddings_size))

    vocab_vector = np.load(FLAGS.vocab_vector_file)

    for index, value in enumerate(vocab_vector):
        embeddings[index] = value
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)


vocab_table = None
vocab_size = None

def get_pred_iterator(batch_size, buffer_size=None, num_threads=4, src_max_len=FLAGS.num_steps):
    input_source_file = tf.placeholder(dtype=tf.string, shape=None, name="input_source_file")
    src_dataset = tf.data.Dataset.from_tensor_slices(input_source_file)
    # 创建词汇表
    global vocab_table, vocab_size
    # 如何将传进来的URL转变成张量
    if vocab_table is None:
        vocab_table = create_vocab_tables(FLAGS.vocab_file)

    if vocab_size is None:
        vocab_size = get_vocab_size(FLAGS.vocab_file)

    if buffer_size is None:
        buffer_size = batch_size * 5

    # 设置src数据集
    src_dataset = src_dataset.map(
        lambda src: (
            tf.string_split([src]).values),
        num_parallel_calls=num_threads
    )
    src_dataset.prefetch(buffer_size)

    src_dataset = src_dataset.map(
        lambda src: (src[:src_max_len]),
        num_parallel_calls=num_threads
    )
    src_dataset.prefetch(buffer_size)

    src_dataset = src_dataset.map(
        # cast(x,dtype,name=None)将x的数据格式转化成dtype.
        lambda src: (tf.cast(vocab_table.lookup(src), tf.int32)),
        num_parallel_calls=num_threads
    )
    src_dataset.prefetch(buffer_size)

    src_dataset = src_dataset.map(
        lambda src: (src, tf.size(src)),
        num_parallel_calls=num_threads
    )
    src_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([src_max_len]),
                           tf.TensorShape([])),
            padding_values=(vocab_size + 1,
                            0))

    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        source_sequence_length=src_seq_len,
        input_source_file=input_source_file,
        input_target_file=None)
