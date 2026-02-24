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

a = 'Hello World'
b = 'bbb'
a = b*2 + a

print(a)