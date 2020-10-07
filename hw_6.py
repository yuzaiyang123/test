import os
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

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

device = torch.device("cuda")

# define input data
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        # initialize each img idx
        self.fnames = []
        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label

    def __len__(self):
        return 200


class Attacker():
    def __init__(self, img_dir, label):
        self.model = models.vgg16(pretrained=True)
        self.model.cuda()
        self.model.eval()
        # rgb三个通道的平均值和方差
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)

        # interpolation=3 对应双线性插值，一种图像增强方法
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            self.normalize
        ])
        self.dataset = Adverdataset('./data/images', label, transform)
        # shuffle 打乱数据后输入模型
        self.loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=1,
                                                  shuffle=False)

    def fgsm_attack(self, image, epsilon, data_grad):
        """

        :param image: 输入图片
        :param epsilon: 往梯度反方向增量的系数
        :param datagrad: 梯度
        :return:
        """
        # 找出 各个dimension的grad取它的符号
        sign_data_grad = data_grad.sign()
        # 当前图片加上这个epsilon
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, epsilon):
        # wrong 代表模型分类结果与真实标签不一致
        # fail加上杂讯后的img仍然被识别正确
        # success加上杂讯后被识别成其他label了
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            # calculate image's each pixel's grad
            data.requires_grad = True
            output = self.model(data)
            # 返回当前列最大的数的序号
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                wrong += 1
                continue
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(
                        self.mean, device=device).view(3, 1, 1)
                    adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                    data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean,
                                                                                                             device=device).view(
                        3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
            final_acc = (fail / (wrong + success + fail))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc

# class Attacker:
#     def __init__(self, img_dir, label):
#         # 讀入預訓練模型 vgg16
#         self.model = models.vgg16(pretrained=True)
#         self.model.cuda()
#         self.model.eval()
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#         # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
#         self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
#         transform = transforms.Compose([
#             transforms.Resize((224, 224), interpolation=3),
#             transforms.ToTensor(),
#             self.normalize
#         ])
#         # 利用 Adverdataset 這個 class 讀取資料
#         self.dataset = Adverdataset('./data/images', label, transform)
#
#         self.loader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=1,
#             shuffle=False)
#
#     # FGSM 攻擊
#     def fgsm_attack(self, image, epsilon, data_grad):
#         # 找出 gradient 的方向
#         sign_data_grad = data_grad.sign()
#         # 將圖片加上 gradient 方向乘上 epsilon 的 noise
#         perturbed_image = image + epsilon * sign_data_grad
#         return perturbed_image
#
#     def attack(self, epsilon):
#         # 存下一些成功攻擊後的圖片 以便之後顯示
#         adv_examples = []
#         wrong, fail, success = 0, 0, 0
#         for (data, target) in self.loader:
#             data, target = data.to(device), target.to(device)
#             data_raw = data;
#             data.requires_grad = True
#             # 將圖片丟入 model 進行測試 得出相對應的 class
#             # output dim 1*1000返回1000个类别的可能性
#             output = self.model(data)
#             # init_pred 返回当前模型判断的类别
#             init_pred = output.max(1, keepdim=True)[1]
#
#             # 如果 class 錯誤 就不進行攻擊
#             if init_pred.item() != target.item():
#                 wrong += 1
#                 continue
#
#             # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
#             loss = F.nll_loss(output, target)
#             self.model.zero_grad()
#             loss.backward()
#             data_grad = data.grad.data
#             perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
#
#             # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class
#             output = self.model(perturbed_data)
#             final_pred = output.max(1, keepdim=True)[1]
#
#             if final_pred.item() == target.item():
#                 # 辨識結果還是正確 攻擊失敗
#                 fail += 1
#             else:
#                 # 辨識結果失敗 攻擊成功
#                 success += 1
#                 # 將攻擊成功的圖片存入
#                 if len(adv_examples) < 5:
#                     adv_ex = perturbed_data * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(
#                         self.mean, device=device).view(3, 1, 1)
#                     adv_ex = adv_ex.squeeze().detach().cpu().numpy()
#                     data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean,
#                                                                                                              device=device).view(
#                         3, 1, 1)
#                     data_raw = data_raw.squeeze().detach().cpu().numpy()
#                     adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
#         final_acc = (fail / (wrong + success + fail))
#
#         print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
#         return adv_examples, final_acc





if __name__ == '__main__':
    # these 200 img's label
    df = pd.read_csv('./data/labels.csv')
    df = df.loc[:, 'TrueLabel'].to_numpy()
    # all sort of label including 1000 categories
    label_name = pd.read_csv('./data/categories.csv')
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    attacker = Attacker('./data/images', df)
    epsilons = [0.1, 0.01]
    accuracies, examples = [], []
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)

