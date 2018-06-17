# coding: utf-8

import tensorflow as tf
import numpy as np
import random
import math
from function import load_data
from sklearn.model_selection import KFold

csv_file_path = r"C:\Users\tamiya\Music\mp3\data.csv"

# パラメーター
num_classes = 2  # クラス数
num_inputs = 168  # 1ステップに入力されるデータ数
num_epoch = 30   #エポック数
batch_size = 10  # バッチサイズ
num_steps = None  # 学習ステップ数
length_sequence = 30  # 系列長
num_nodes = 256  # ノード数
num_data = None  # 各クラスの学習用データ数



random.seed(0)

test_accuracy_list = []

x_data, t_data = load_data(csv_file_path)
x_data = np.array(x_data)
t_data = np.array(t_data)
perm = np.random.permutation(len(x_data))
x_data = x_data[perm]
t_data = t_data[perm]
#print("yes: %d", np.sum(t_data == 1))
#print("no: %d",  np.sum(t_data == 0))

# データを10分割する
n_fold = 10
k_fold = KFold(n_fold, shuffle=True)

# モデルの構築
x = tf.placeholder(tf.float32, [None, length_sequence, num_inputs])  # 入力データ
t = tf.placeholder(tf.int32, [None])  # 教師データ
t_on_hot = tf.one_hot(t, depth=num_classes, dtype=tf.float32)  # 1-of-Kベクトル
cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=num_nodes, activation=tf.nn.sigmoid)  # 中間層のセル

# RNNに入力およびセル設定する
outputs1, states1 = tf.nn.dynamic_rnn(cell=cell1, inputs=x, dtype=tf.float32, time_major=False, scope="lstm1")

# [ミニバッチサイズ,系列長,出力数]→[系列長,ミニバッチサイズ,出力数]
#outputs = tf.transpose(outputs1, perm=[1, 0, 2])

outputs = tf.reshape(outputs1, [-1, length_sequence * num_nodes])
#w = tf.Variable(tf.random_normal([num_nodes, num_classes], stddev=0.01))
w = tf.Variable(tf.random_normal([length_sequence * num_nodes, num_classes], stddev=0.01))
b = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(outputs, w) + b  # 出力層
pred = tf.nn.softmax(logits)  # ソフトマックス

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)  # 誤差関数
train_step = tf.train.AdamOptimizer().minimize(loss)  # 学習アルゴリズム

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(t_on_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 精度

fold_count = 0
for train_idx, test_idx in k_fold.split(x_data, t_data):
    x_train, t_train = x_data[train_idx], t_data[train_idx]  # 学習用データセット
    x_test, t_test = x_data[test_idx], t_data[test_idx]  # テスト用データセット
    num_data = len(x_train)
    num_steps = int (num_epoch * (num_data / batch_size))
    fold_count += 1

    # 学習の実行
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(num_steps):
        cycle = int(num_data / batch_size)
        begin = int(batch_size * (i % cycle))
        end = begin + batch_size
        x_batch, t_batch = x_train[begin:end], t_train[begin:end]
        sess.run(train_step, feed_dict={x: x_batch, t: t_batch})
        if i % math.floor(num_data/batch_size) == 0:
            loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: x_train, t: t_train})
            loss_test_, acc_test_ = sess.run([loss, accuracy], feed_dict={x: x_test, t: t_test})
            print("fold%d epoch %d: train_loss:%f, train_acc:%f test_loss:%f, test_acc:%f" % (fold_count, i / (num_data / batch_size), loss_, acc_, loss_test_, acc_test_))
            data_set = list(zip(x_train, t_train))
            random.shuffle(data_set)
            x_train, t_train = zip(*data_set)

    test_accuracy_list.append(sess.run(accuracy, feed_dict={x: x_test, t: t_test}))
    #print(sess.run(accuracy, feed_dict={x: x_test, t: t_test}))
    sess.close()

test_accuracy_list = np.array(test_accuracy_list)
print(test_accuracy_list)
print(test_accuracy_list.mean(axis=0))
