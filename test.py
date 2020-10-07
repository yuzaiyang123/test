import os
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np
import math
import torch
# Loss function
import torch.nn.functional as F
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt

# device = torch.device("cuda")
# df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
#      index=['cobra', 'viper', 'sidewinder'],
#      columns=['max_speed', 'shield'])
# print(df)
# df = df.loc[:, 'max_speed']
# print(df)
# if __name__ == '__main__':
#     df = pd.read_csv("./data/labels.csv")
#     df = df.loc[:, 'TrueLabel'].to_numpy()
#     df = pd.read_csv('./data/labels.csv')
#     df = df[:, 'TrueLabel'].to_numpy()
#     print(df)
# a = [2,4,7,1]
# a = torch.tensor(a)
# print(a)
# a=a.view(1,-1)
# a = a.max(1)[1]
# print(a)
# for i in range(10):
#      a.append("{:03d}".format(i))
# print(a)
# input is of size N x C = 3 x 5

# each element in target has to have 0 <= value < C
output = torch.randn(1, 3)  # 网络输出
target = torch.ones(1, dtype=torch.long).random_(3)  # 真实标签
print(output)
print(target)
loss = F.nll_loss(output, target)
print(loss)