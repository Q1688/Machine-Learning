#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Author : lianggq
# @Time  : 2019/7/19 10:24
# @FileName: tools.py
import numpy as np
import jieba

jieba.load_userdict('../stopword/myjieba.txt')
from tensorflow import keras as kr
from article_categories.config.config import FLAGS


def read_data(filename):
    """
    对文件数据中标签和内容进行分离
    :param filename: 数据文件名
    :return:标签和内容列表
    """
    try:
        label_list = []
        data_list = []

        with open(filename, 'r', encoding='utf-8') as r:
            lines = r.readlines()
            for line in lines:
                line = line.strip()
                line = line.split('\t', 1)
                label = line[0]
                label_list.append(label)
                data_list.append(line[1])
        return label_list, data_list

    except:
        print('读取训练数据异常')


def build_stop_word(stopword_filename):
    """
    引用停用词
    :param stopword_filename:停用词文件名
    :return:停用词列表
    """
    try:
        with open(stopword_filename, 'r', encoding='utf-8') as r:
            stopword = r.read()
            stop_word = stopword.split('\n')
            stop_list = [word for word in stop_word]
            return stop_list

    except:
        print('读取停用词异常')


def write_data_clean(labels, contents, label_filename, content_filename):
    """
    对数据清洗之后写入文件，并判断数据和标签的长度是否相同
    :param labels: 标签列表
    :param contents: 数据列表
    :param label_filename: 写入标签的文件
    :param content_filename: 写入清洗数据的文件
    :return:
    """
    # 引用停用分词
    stop_list = build_stop_word('../stopword/stopword.txt')

    # 标签文件
    with open(label_filename, 'w+', encoding='utf-8') as w:
        for label in labels:
            w.write(label + '\n')

    # 数据文件
    with open(content_filename, 'w+', encoding='utf-8') as f:
        for content in contents:
            word_list = []
            document_cut = jieba.cut(content, cut_all=False)
            for word in document_cut:
                if word not in stop_list and word is not ' ':
                    word_list.append(word)
            f.write(str(word_list).replace('\'', '').strip('[').strip(']') + '\n')
    print('数据清洗完毕')


def build_vocab(data, vocab_file, vector_word_npz, vocab_size=100000):
    """构建词汇表和词汇长度表"""
    word_count = {}

    # 如何构建字典
    with open(data, 'r', encoding='utf-8') as r:
        for line in r.readlines():
            line = line.split(',')
            for word in line:
                word_count[word.strip()] = word_count.get(word.strip(), 0) + 1

    with open(vocab_file, 'w+', encoding='utf-8') as w:
        # 对字典进行排序，统计词频前,获取前10000个元素作为词表
        word_list = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for word in word_list[:vocab_size]:
            w.write(word[0] + '\n')

    # 构建向量表
    np.save(vector_word_npz, np.array(word_list[:vocab_size]))


def buile_onehot(vocab_filename, clean_label, clean_content):
    # {vocab:index}
    vocab, vocab_id = bulid_vocab_id(vocab_filename)
    # {类别：index}
    label_to_id = build_class_id(clean_label)

    words_to_id = []
    labels_to_id = []

    content_list = read_clean_content(clean_content)
    label_list = read_clean_label(clean_label)

    # 判断标签和内容的行数是否相等
    if len(content_list) != len(label_list):
        raise ValueError('line number was different with write source and target in files')

    for i in range(len(content_list)):
        words_to_id.append([vocab_id[word.strip()] for word in content_list[i].split(',') if word.strip() in vocab_id])
        labels_to_id.append(label_to_id[label_list[i].strip('\n')])

    # 使用keras模块提供的pad_sequence来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(words_to_id, maxlen=FLAGS.seq_length)
    y_pad = kr.utils.to_categorical(labels_to_id, num_classes=len(label_to_id))  # 将y转换成one-hot向量
    return x_pad, y_pad


def bulid_vocab_id(vocab_filename):
    '''{vocab:index}'''
    with open(vocab_filename, 'r', encoding='utf-8') as r:
        vocab = r.readlines()
        vocab = list(set([word.strip() for word in vocab]))
        vocab_id = dict(zip(vocab, range(len(vocab))))
    return vocab, vocab_id


def build_class_id(label_filename):
    '''{类别：index}'''
    label_dir = {}
    with open(label_filename, 'r', encoding='utf-8') as r:
        for label in r.readlines():
            label_dir[label.strip('\n')] = label_dir.get(label.strip('\n'), 0) + 1

    label_list = [labels for labels in label_dir.keys()]
    label_to_id = dict(zip(label_list, range(len(label_list))))
    return label_to_id


def read_clean_content(clean_content):
    content_list = []
    with open(clean_content, 'r', encoding='utf-8') as r:
        for content in r.readlines():
            content_list.append(content)

    return content_list


def read_clean_label(clean_label):
    label_list = []
    with open(clean_label, 'r', encoding='utf-8') as r:
        for label in r.readlines():
            label_list.append(label)

    return label_list


# 生成批次数据
def batch_iter(x, y, batch_size):
    num_batch = int((len(x) - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(len(x)))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, len(x))

        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
