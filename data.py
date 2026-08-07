"""
data.py —— 数据集准备 page73-74 页
定义 transforms 预处理流程，加载 CIFAR10，
并封装成 trainloader / testloader 供训练使用。
"""

import torch
from torchvision import datasets, transforms

# Train set预处理
# Compose 按照列表顺序依次处理图片
# 随机裁剪->随机翻转->转化为张量->标准化 （前两步均为数据增强）
train_transforms = transforms.Compose([
    #填充图片到40*40，进行随机裁剪 -->减少死记风险1
    transforms.RandomCrop(32, padding=4), 
    #默认0.5的概率水平翻转(此处介于次数据集的特殊性，水平翻转并不影响识别物品类别)
    transforms.RandomHorizontalFlip(),  

    # 转化为Tensor
    # -调整维度格式为[batch, channel, height, width]
    # -转换像素为浮点数（0～225->0.0~1.0)->缩小了数值
    transforms.ToTensor(),  

    # 标准化 每一格像素 ->（原值 - 该通道均值）÷ 该通道标准差
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  
])


# Test set预处理
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


# 创建CIFAR-10数据集
trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transforms)
testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transforms)
# root='./data'：数据保存在当前工作目录的 data 文件夹。
# train=True：加载训练集；train=False：加载测试集
# download=True：本地没有数据时自动下载。
# transform不是在创建trainset时一次性处理所有照片，而是在训练期间执行具体操作时再处理
# ->一张原始照片再不同的epoch中可能成为不同的训练输入->训练时样本版本增多了->扩大多样性
# -->减少死记风险2


# 打包成DataLoader，准备训练
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=4)

# trainloader->每次从trainset中取出128个数据（为一个batch）打包交给模型
# batch = inputs, targets = next(iter(trainloader))
# inputs.shape = [128,3,32,32]
# targets.shape = [128] (ps targets中的元素为对应图片的类型编号0-9)

# trainset-Shuffle=True->每个epoch重新打乱样本顺序，避免死记
# num_workers=4 同时启动四个子进程执行相同工作（PyTorch封装好了的...
