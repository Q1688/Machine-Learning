#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Author : lianggq
# @Time  : 2019/7/23 11:43
# @FileName: train.py
import gc

from sklearn import metrics
from pandas.tests.extension.numpy_.test_numpy_nested import np

from article_categories.utils.cnn import CNN
from article_categories.utils.tools import buile_onehot, batch_iter, build_class_id
from article_categories.config.config import FLAGS
import os
import tensorflow as tf


def feed_data(x_batch, y_batch, dropout_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_prob: dropout_prob
    }
    return feed_dict


def evaluate(sess, x, y):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x)
    batch_eval = batch_iter(x, y, FLAGS.batch_size)

    total_acc = 0.0
    total_loss = 0.0

    for x_batch, y_batch in batch_eval:
        feed_dict = feed_data(x_batch, y_batch, FLAGS.dropout_keep_prob)
        # 算出来的loss和acc是在这一批次数据上的均值
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss = total_loss + loss * len(x_batch)
        total_acc = total_acc + acc * len(x_batch)

    return total_loss / data_len, total_acc / data_len


def train_model(x_train, y_train, vx_pad, vy_pad):
    # 模型保存的位置
    save_dir = "./model/"
    save_path = os.path.join(save_dir, "best_validation")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()

    # 配置Tensorboard
    # 收集训练过程中的loss和acc
    tensorboard_dir = "./parameter_file/"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    # 指定一个文件用来保存图
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 总的训练批次
    total_batch = 1
    # 最佳测试集的准确率
    best_test_acc = 0.0
    # 上一次验证集准确率提升时的训练批次
    last_improved_batch = 0
    # 若超过800轮次没有提升，则提前结束训练
    require_improved = 1000
    # 标识是否需要提前结束训练
    flag = False

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch in range(FLAGS.num_epochs):
            batch_train = batch_iter(x_train, y_train, FLAGS.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, FLAGS.dropout_keep_prob)

                # 每多少轮次将数据写入parameter_file
                if total_batch % FLAGS.save_per_batch == 0:
                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(summary, total_batch)

                # 每多少轮输出在训练集和验证集上的损失及准确率
                if total_batch % FLAGS.print_per_batch == 0:
                    # 训练集损失和准确率
                    feed_dict[model.dropout_prob] = 1.0
                    train_loss, train_acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)

                    # 用验证集对损失和准确率进行评估
                    val_loss, val_acc = evaluate(sess, vx_pad, vy_pad)

                    # 保存最佳模型
                    if val_acc > best_test_acc:
                        best_test_acc = val_acc
                        saver.save(sess, save_path)
                        last_improved_batch = total_batch

                    msg = "Iter:{0:>4},Train loss:{1:>6.2}, Train accuracy:{2:>6.2%}, Val loss:{3:>6.2}, Val accuracy:{4:>6.2%}"
                    print(msg.format(total_batch, train_loss, train_acc, val_loss, val_acc))

                # 优化
                sess.run(model.optim, feed_dict=feed_dict)
                total_batch = total_batch + 1

                del x_batch
                del y_batch
                gc.collect()

                if total_batch - last_improved_batch == require_improved:
                    # 验证集准确率不提升时，停止训练
                    print("No improvement for a long time, auto-stopping..................")
                    flag = True
                    break
            if flag:
                break


def test_model(label_list, x_pad, y_pad):
    # 模型的路径
    save_dir = "./model/"
    save_path = os.path.join(save_dir, "best_validation")
    # 加载已经持久化的图
    saver = tf.train.import_meta_graph("./model/best_validation.meta")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载之前保存的模型
        saver.restore(sess, save_path=save_path)
        test_loss, test_acc = evaluate(sess, x_pad, y_pad)
        print("Test loss:%0.2f, Test accuracy:%0.2f" % (test_loss, test_acc))

        batch_size = FLAGS.batch_size
        data_len = len(x_pad)
        num_batch = int((data_len - 1) / batch_size) + 1

        y_test = np.argmax(y_pad, 1)
        y_pred = np.zeros(shape=len(x_pad), dtype=np.int32)  # 保存预测结果
        for i in range(num_batch):  # 逐批次处理
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            feed_dict = {
                model.input_x: x_pad[start_id:end_id],
                model.dropout_prob: 0.9
            }
            y_pred[start_id:end_id] = sess.run(model.y_pred, feed_dict=feed_dict)

        # 此处也可以写成像训练集数据一样的准确率
        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test, y_pred, target_names=label_list))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(cm)


if __name__ == '__main__':
    model = CNN()

    # 构建训练集向量
    x_pad, y_pad = buile_onehot(FLAGS.vocab_filename, FLAGS.train_label_file, FLAGS.train_content_file)
    # 构建验证集向量
    vx_pad, vy_pad = buile_onehot(FLAGS.vocab_filename, FLAGS.val_label_file, FLAGS.val_content_file)
    # 对数据进行训练模型构建
    train_model(x_pad, y_pad, vx_pad, vy_pad)

    print('训练模型完毕，开始预测模型')
    label_list = [label for label in build_class_id(FLAGS.train_label_file).keys()]
    x_pad, y_pad = buile_onehot(FLAGS.vocab_filename, FLAGS.test_label_file, FLAGS.test_content_file)
    test_model(label_list, x_pad, y_pad)
