# coding: utf-8
from pickletools import optimize
import sys
sys.path.append('..')
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from trainer import RNNTrainer
from simple_rnnlm import SimpleRnnlm
from common.optimizer import SGD
from dataset import ptb
import pandas as pd
from common.util import custom_preprocess

def main():

    # データの用意 (1000, 2)
    df = pd.read_csv("../data/sin_data.csv")
    print(df.shape)

    # train_X = df["ymd"].to_numpy().reshape(-1, 1)
    # train_T = df["Stock_1"].to_numpy()

    # test_X = df["ymd"].to_numpy().reshape(-1, 1)
    # test_T = df["Stock_1"].to_numpy()

    # print(df.describe())
    # print(df.info())
    # print(df.shape) # (365, 6)
    # print(df.columns)

    # fig_col = 3
    # fig_index = 2
    # fig_num = 1
    # col_num = 1

    # fig = plt.figure(figsize=(12, 8))

    # for i in range(1, fig_index + 1):
    #     for c in range(1, fig_col + 1):

    #         if fig_num >= (fig_col * fig_index):
    #             break
    #         ax1 = fig.add_subplot(fig_index, fig_col, fig_num)
    #         ax1.set_title(f"{df.columns[col_num]}")
    #         ax1.set_xlabel(f"{df.columns[0]}")
    #         ax1.set_ylabel(f"{df.columns[col_num]}")
    #         ax1.plot(df["ymd"], df.iloc[:, col_num])

    #         fig_num += 1
    #         col_num += 1

    # plt.savefig("original.png")

    # ハイパーパラメータの設定
    batch_size = 10
    time_size = 5

    # corpus, word_to_id, id_to_word = custom_preprocess(train_X.flatten())

    # vocab_size = int(max(corpus) + 1)
    # wordvec_size = 100
    # hidden1_size = 100
    # hidden2_size = 1

    # model = RNNRegressor(vocab_size, wordvec_size, hidden1_size, hidden2_size)
    # optimizer = SGD()

    # trainer = RNNTrainer(model, optimizer)
    # trainer.fit(train_X, train_T)

    # file_name = "train_result"
    # trainer.plot(file_name)

    # # テスト結果
    # test_result = trainer.predict(test_X.flatten())
    # mse = len(1 / test_X.flatten.shape[0] * 1/2 * (test_result, test_T))
    # print(f"平均二乗誤差 {mse}")


if __name__ == "__main__":
    main()