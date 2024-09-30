import torch
import timm  # 用于加载 EfficientNet 模型
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from MahjongDataset import MahjongDataset



# 定义模型
model = timm.create_model('efficientnet_b0', pretrained=True)  # 加载预训练的 EfficientNet-B0
num_classes = 42  # 总共有 42 类麻将牌
model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # 修改最后的分类层以适应类别数

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



