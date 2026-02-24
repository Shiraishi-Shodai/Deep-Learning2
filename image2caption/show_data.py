import kagglehub
import polars as pl
from matplotlib import pyplot as plt
import csv
from pathlib import Path
import cv2
import sys
sys.path.append('..')
from common.np import *
from torch.nn import functional as F
import torch

# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)

# =================================
# テキストファイルをcsvファイルに変換
# =================================
text_file_path = Path(rf'{path}\captions.txt')
csv_file_path = Path(rf'{path}\captions.csv')

with open(text_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(csv_file_path, 'w', newline="", encoding='utf-8') as f:
    writer = csv.writer(f)

    for line in lines:
        row = line.strip().split(',', 1)
        writer.writerow(row)

# =================================
# csvファイルのキャプションに<start> <end>を追加
# =================================
csv_dataframe = pl.read_csv(csv_file_path)
START_TOKEN = '<start>'
LAST_TOKEN = '<end>'
PAD_TOKEN = '<pad>'

def start_last_token_check(df, START_TOKEN, LAST_TOKEN, PAD_TOKEN):

    df = df.with_columns(
        pl.when(~pl.col("caption").str.starts_with(START_TOKEN))
        .then(pl.lit(START_TOKEN) + " " + pl.col("caption"))
        .otherwise(pl.col("caption"))
        .alias("caption")
    )

    df = df.with_columns(
        pl.when(~pl.col("caption").str.ends_with(LAST_TOKEN))
        .then(pl.col("caption") + " " + pl.lit(LAST_TOKEN))
        .otherwise(pl.col("caption"))
        .alias("caption")
    )

    df = df.with_columns(
        pl.col("caption").str.split(" ").alias("tokens")
    )

    max_len = df.select(pl.col("tokens").list.len().max()).item()

    df = df.with_columns(
        pl.when(pl.col("tokens").list.len() < max_len)
        .then(
            pl.col("tokens").list.concat(
                pl.lit(PAD_TOKEN).repeat_by(
                    max_len - pl.col("tokens").list.len()
                )
            )
        )
        .otherwise(pl.col("tokens"))
        .alias("tokens")
    )

    # ← CSV保存用に文字列へ戻す
    df = df.with_columns(
        pl.col("tokens").list.join(" ").alias("caption")
    ).drop("tokens")

    return df

csv_dataframe = start_last_token_check(csv_dataframe, START_TOKEN, LAST_TOKEN, PAD_TOKEN)
csv_dataframe.write_csv(csv_file_path)

# =================================
# read csv data
# =================================
# data = pl.read_csv(str(csv_file_path))
# print(data.describe())

# =================================
# 画像を一枚表示
# =================================
# fisrt_img_name = data[0, 0]
# first_img_src = Path(rf'{path}\Images\{fisrt_img_name}')
# fisrt_captions = data.filter(pl.col('image') == fisrt_img_name)["caption"]

# # print(fisrt_img_name)
# print(fisrt_captions, type(fisrt_captions))

# fig = plt.figure()
# ax = plt.subplot(1, 1, 1)
# img = cv2.imread(first_img_src)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 各キャプションを表示する位置を取得
# caption_size = len(fisrt_captions)
# base_position_bottom = -0.025
# # start_position_bottom = -0.1
# caption_position_bottom = np.arange(base_position_bottom, base_position_bottom*caption_size, base_position_bottom)

# print(caption_position_bottom)
# print(type(caption_position_bottom))

# for caption, position_bottom in zip(fisrt_captions, caption_position_bottom):
#     ax.text(
#         0.5,
#         position_bottom,
#         caption,
#         transform=ax.transAxes,
#         ha="center"
#     )

# ax.imshow(img)
# ax.set_axis_off()
# plt.show()

# =================================
# ワンホットベクトル実験
# =================================
# word_size = 5
# a = torch.arange(0, word_size)
# one_hot_vec = F.one_hot(a)
# print(a)
# print(one_hot_vec)

# =================================
# 単語IDを辞書化
# =================================
dict_path = rf'{path}\word_dict.csv'
# word_series = data["caption"]

def make_dict(word_series, dict_path):
    """単語の辞書を作成する
    """
    word_dict = {"id": [] , "word" : []}

    for caption in word_series:
        words = caption.split()
        for word in words:
            if word not in word_dict["word"]:
                new_id = next(reversed(word_dict["id"])) + 1 if len(word_dict["id"]) > 0 else 0
                word_dict["id"].append(new_id)
                word_dict["word"].append(word)
    
    word_dataframe = pl.DataFrame(word_dict)
    word_dataframe.write_csv(dict_path)

# make_dict(word_series, dict_path)

# =================================
# one-hotベクトル化
# =================================
# word_dataframe = pl.read_csv(dict_path)
# print(word_dataframe)
# last_id = next(reversed(word_dataframe["id"]))
# print(F.one_hot(torch.arange(0, last_id + 1)))
# print(F.one_hot(torch.arange(0, last_id + 1)).shape)
