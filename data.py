"""
data.py —— 数据集准备
参照课件 73-74 页：定义 transforms 预处理流程，加载 CIFAR10，
并封装成 trainloader / testloader 供训练使用。
"""

import torch
from torchvision import datasets, transforms

# ------------------ 1. 定义预处理流程 ------------------
# 训练集：随机裁剪 + 随机水平翻转做数据增强，再转 Tensor 并归一化
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# 测试集：不做随机增强，只做归一化，保证评估结果稳定可复现
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# ------------------ 2. 加载 CIFAR10 数据集 ------------------
trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transforms)
testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transforms)

# ------------------ 3. 封装成 DataLoader ------------------
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=4)
