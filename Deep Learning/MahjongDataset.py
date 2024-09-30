import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 自定义数据集类
class MahjongDataset(Dataset):
    def __init__(self, csv_file, label_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): 包含图片文件名和标签的 CSV 文件路径。
            label_file (str): 包含标签名和类别索引的 CSV 文件路径。
            root_dir (str): 所有图片文件所在的根目录。
            transform (callable, optional): 数据预处理操作。
        """
        self.data_frame = pd.read_csv(csv_file)  # 读取包含图片名和标签的 CSV 文件
        self.label_map = pd.read_csv(label_file)  # 读取包含标签映射的 CSV 文件

        # 创建一个从 label-name 到 label-index 的映射
        self.label_dict = dict(zip(self.label_map['label-name'], self.label_map['label-index']))

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)  # 数据集的长度

    def __getitem__(self, idx):
        # 读取图片的文件名和标签名称
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)  # 使用 PIL 打开图片

        # 获取图片的标签名称
        label_name = self.data_frame.iloc[idx, 2]

        # 将标签名称映射为标签索引
        label = self.label_dict[label_name]

        # 数据预处理（如果有）
        if self.transform:
            image = self.transform(image)

        return image, label  # 返回图片和对应的标签


# 数据预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])

# CSV 文件路径和图像根目录路径
csv_file = 'D:/demo_exe/majianglizhi/majiangOCR/data_pic/mahjong-dataset-master/tiles-data/data.csv'
label_file = 'D:/demo_exe/majianglizhi/majiangOCR/data_pic/mahjong-dataset-master/tiles-data/label.csv'
root_dir = 'D:/demo_exe/majianglizhi/majiangOCR/data_pic/mahjong-dataset-master/tiles-resized'

# 初始化数据集
mahjong_dataset = MahjongDataset(csv_file=csv_file, label_file=label_file, root_dir=root_dir, transform=transform)


########################################################################################################################################################
#测试数据
########################################################################################################################################################
# # 使用 DataLoader 加载数据
# train_loader = DataLoader(mahjong_dataset, batch_size=32, shuffle=True)

# # 测试数据加载
# for images, labels in train_loader:
#     print(images.shape, labels.shape)
#     break
# for images, labels in train_loader:
#     print("Labels:", labels)  # 打印每个批次中的标签
#     break
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 显示图像的函数
# def imshow(img):
#     img = img / 2 + 0.5  # 反归一化
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # 从 DataLoader 中获取一个批次的图像和标签
# dataiter = iter(train_loader)
# images, labels = next(dataiter)  # 使用 next() 来获取下一个批次
#
# # 显示其中一张图片
# imshow(images[0])
# print("Label:", labels[0].item())

