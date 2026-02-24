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


# =================================
# read csv data
# =================================
path = r"C:\Users\siran\.cache\kagglehub\datasets\adityajn105\flickr8k\versions\1"
data_file_path = Path(rf'{path}\captions.csv')
dict_path = Path(rf'{path}\word_dict.csv')
data = pl.read_csv(str(data_file_path))
print(data.describe())

# =================================
# one-hotベクトル化
# =================================
word_dataframe = pl.read_csv(str(dict_path))
print(word_dataframe)
# print(onehot_vec)
# print(onehot_vec.shape)

# idx = torch.where(onehot_vec[-1] == 1)
# print(idx, type(idx))

# word = word_dataframe[idx, "word"]
# print(word, type(word))

# =================================
# DataLoaderの取得
# =================================
