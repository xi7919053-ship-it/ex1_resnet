"""
ResNet.py定义模型结构
好抽象啊啊啊啊啊
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    
    expansion = 1  # BasicBlock 输出通道数 = planes * expansion

    # BasicBlock定义（可反复使用）
    # 第一层：3*3Conv+BN；第二层：3*3Conv+BN
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        '''
        Convolutional layer：用许多小型fliters在图片上滑动，寻找局部特征
        *3 卷积核们/kernels/fliters 每次移动一个位置（stride=1）
        padding=1在图片四周补一圈像素
        输出feature maps

        BatchNorm 会对一批数据的特征进行标准化和调整，使数据分布更稳定
        '''

        # in_planes-输入通道数
        # planes-输出通道数
        # stride卷积移动步长

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        # shortcut什么都不做：shortcut(x)=x
        # 因为Tensor形状没变，可以直接按照顺序相叠加
        self.shortcut = nn.Sequential() 

        # shortcut卷积
        # 当宽高发生变化（stride变化）或者输入输出通道数不一致时
        # shortcut 需要用 1x1 卷积把维度对齐，否则无法相加
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # 向前传播
    def forward(self, x):
        
        # x是输入BasicBlock的一批特征图
        
        # ReLU(x) = max(0,x) 引入非线性变化
        # 理解：输出<0-->0; 输出>=0-->不变
        
        # 第一层卷积
        out = F.relu(self.bn1(self.conv1(x)))  # F: torch.nn.functional
        # 第二层卷积
        # 卷积后先将结果与shortcut相加，再执行ReLU
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # 残差连接：捷径分支 + 主干分支相加
        out = F.relu(out)   


        return out


# ResNet主体的初始化
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # e.g. [B,3,32,32]-> [B,64,32,32]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4 个 stage，每个 stage 由若干个 BasicBlock 堆叠而成
        # 空间尺寸逐渐减小：32 → 16 → 8 → 4
        # 特征通道逐渐增加：64 → 128 → 256 → 512
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # nn.Linear(输入特征数, 输出特征数)
        # 512个输入特征->全连接层->10个分类分数
        # y=wx+b
        # x：512个输入特征 w：需要训练的权重 b：需要训练的偏置 y：10个输出分数
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # 创建stage
    '''
    ???
    '''
    def _make_layer(self, block, planes, num_blocks, stride):
        # 每个 stage 第一个 block 负责下采样(stride可能为2)，其余 stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # [B, 64, 32, 32]
        out = self.layer1(out)                    # [B, 64, 32, 32]
        out = self.layer2(out)                    # [B, 128, 16, 16]
        out = self.layer3(out)                    # [B, 256, 8, 8]
        out = self.layer4(out)                    # [B, 512, 4, 4]
        out = F.adaptive_avg_pool2d(out, 1)        # 全局平均池化 -> [B, 512, 1, 1]
        out = out.view(out.size(0), -1)            # 展平 -> [B, 512]
        out = self.linear(out)                     # 全连接分类 -> [B, 10]
        return out


def ResNet18(num_classes=10):
    """
    ResNet18 = BasicBlock 堆叠 [2,2,2,2]，
    每个 BasicBlock 有2层卷积，4个stage共8个block，8*2=16层卷积 + 第一层conv + 全连接
    2*(2+2+2+2)+2 = 18
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# 简单自测：人为输入一个 batch 的假数据，检查输出形状是否正确
if __name__ == '__main__':
    net = ResNet18()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print('输出形状:', y.shape)  # 期望: torch.Size([2, 10])
