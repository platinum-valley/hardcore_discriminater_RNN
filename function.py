# coding: utf-8

import re
import os
import pickle
from wav2mfcc import mp3_to_vector


def load_data(csv_path):
    """
    csvファイルから音声ファイル情報を取得し，音声ファイルの特徴量(入力データ)と正解クラス(出力データ)を得る
    pickleファイルに取得した入力データ(x_data.pickle)と出力データ(y_data.pickle)を保存する
    pickleファイルがあれば，そこからデータを読む

    :param csv_path: 音声ファイル情報を記述したcsvファイルのパス
    """
    x_pickle_path = "x_data.pickle"
    y_pickle_path = "y_data.pickle"

    if os.path.exists(x_pickle_path):
        with open(x_pickle_path, "rb") as xf:
            x_data = pickle.load(xf)
        with open(y_pickle_path, "rb") as yf:
            y_data = pickle.load(yf)
        return x_data, y_data

    # pickleファイルがない場合

    x_data = []
    y_data = []
    current_path = os.getcwd()
    with open(csv_path, "r", encoding="utf-16") as csv:
        csv_tuples = csv.readlines()

    for tuple in csv_tuples:
        cell = re.split(";", tuple, flags=re.IGNORECASE)
        mp3_directory_path = cell[8]
        mp3_file_name = cell[9]
        x_data.append(mp3_to_vector(mp3_directory_path, mp3_file_name))
        if cell[11][:-1] == "Yes":
            y_data.append(1)
        else:
            y_data.append(0)
        print("load: " + mp3_file_name )

    with open(current_path + "/" + x_pickle_path, "wb") as xf:
        pickle.dump(x_data, xf)

    with open(current_path + "/" + y_pickle_path, "wb") as yf:
        pickle.dump(y_data, yf)

    return x_data, y_data




