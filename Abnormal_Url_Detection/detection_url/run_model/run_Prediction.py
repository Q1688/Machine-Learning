#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time  : 2019/5/27 16:24
# @FileName: run_Prediction.py

import tensorflow as tf
import collections
from Abnormal_Url_Detection.detection_url.config.config import FLAGS
from Abnormal_Url_Detection.detection_url.utils.utils import get_pred_iterator
import datetime
from Abnormal_Url_Detection.detection_url.tools.loger import logger


class ConvertedModel(collections.namedtuple("ConvertedModel",
                                            ("source",
                                             "source_sequence_length",
                                             "y_pred",
                                             ))):
    pass


def save2pb(checkpoint_file, graph_file):
    with tf.Session() as sess:
        # 构造网络图
        saver = tf.train.import_meta_graph(graph_file)
        # 恢复以前保存的变量
        saver.restore(sess, checkpoint_file)
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            ["y_pred"],
        )
        # 保存异常模型检测.【GFile类似于python中的open】
        with tf.gfile.GFile(FLAGS.model_pb_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())


def load_pb(model_path):
    # tf.gfile.FastGFile 实现对图片的读取,‘r’:UTF-8编码; ‘rb’:非UTF-8编码
    with tf.gfile.FastGFile(model_path, "rb") as f:
        # 新建GraphDef文件，用于临时载入模型中的图
        graph_def = tf.GraphDef()
        # GraphDef加载模型中的图
        graph_def.ParseFromString(f.read())

    # 在空白图中加载GraphDef中的图
    y_pred, source, source_sequence_length = tf.import_graph_def(
        graph_def,
        return_elements=["y_pred:0", "IteratorGetNext:0", "IteratorGetNext:2"]
    )
    return ConvertedModel(
        source=source,
        source_sequence_length=source_sequence_length,
        y_pred=y_pred)


def pred_txt_data_by_pb(cm, iterator, urls):
    """
    通过生成的模型文件来预测
    :return:
    """
    with tf.Session() as sess:
        tf.tables_initializer().run()

        sess.run(iterator.initializer, feed_dict={iterator.input_source_file: [urls]})  # 更换测试文件的值
        # 初始化预测值
        pred_value = []
        try:
            s, ssl = sess.run([iterator.source, iterator.source_sequence_length])
            pv = sess.run(cm.y_pred, feed_dict={cm.source: s,
                                                cm.source_sequence_length: ssl})
            pred_value.extend(pv)
        except tf.errors.OutOfRangeError as reason:
            logger.error(reason)
        return pred_value


def model():
    save2pb("../model_filt/points-100", "../model_filt/points-100.meta")


if __name__ == '__main__':
    # model()  # 此步骤执行一次就好,创建模型二进制文件之后就可以注释
    starttime = datetime.datetime.now()
    cm = load_pb(FLAGS.model_pb_file)
    iterator = get_pred_iterator(batch_size=FLAGS.batch_size)
    with open('../read_file.txt', 'r', encoding='utf-8') as f:
        for line in f:
            strs = " ".join(line)
            lable = pred_txt_data_by_pb(cm, iterator, strs)
            print(lable, line)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
