"""
train.py
五步法：
  1. 准备数据集   (由 data.py 提供，这里直接传入 trainloader/testloader)
  2. 定义网络结构 (由 models 提供，这里传入已实例化的 model)
  3. 定义损失函数 (交叉熵损失，分类任务对应课件 loss function 部分)
  4. 定义优化算法 (SGD + 学习率衰减)
  5. 迭代训练     (前向传播 -> 计算loss -> 反向传播 -> 更新参数)
"""

import time # 统计每个 epoch 花费的时间
import torch # Tensor、设备、梯度等基础功能
import torch.nn as nn # 神经网络组件和损失函数
import torch.optim as optim # SGD 等优化器和学习率调整器


def get_best_device(prefer_gpu=True):
    """
    自动选择最合适的训练设备：
    优先级：CUDA (NVIDIA显卡) > MPS (Mac M系列芯片自带GPU) > CPU
    """
    if not prefer_gpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Trainer:
    def __init__(self, model_name, model, train_on_gpu=False):
        self.model_name = model_name
        self.device = get_best_device(prefer_gpu=train_on_gpu)
        self.model = model.to(self.device) # 把模型放到设备上

        # 定义损失函数
        # criterion 用于衡量模型预测错了多少
        # loss = self.criterion(outputs, targets)
        self.criterion = nn.CrossEntropyLoss()

        print('训练设备: {}'.format(self.device))

    def _build_optimizer(self, lr):  #_表示供class内部使用
        # 定义优化算法
        # 梯度下降法 w <- w - eta * dL/dw
        # CrossEntropyLoss 告诉模型“错了多少”，SGD 根据这个错误去调整卷积核和全连接层的权重
        # SGD-Stochastic Gradient Descent
        # + momentum （有动量：保留之前的运动趋势，方向更稳定）
        # + weight_decay（避免模型参数变得过大） 加速收敛、防止过拟合
        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=5e-4)
        # 学习率调整器
        # scheduler 用来在训练过程中自动调整学习率。
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        # 前期       0.2
        # 第100轮后  0.02
        # 第150轮后  0.002 学习率降低->步伐调小
        return optimizer, scheduler

    def _train_one_epoch(self, trainloader, optimizer):
        self.model.train() # 把模型切换到训练模式

        running_loss = 0.0 # 所有样本的累计损失
        correct = 0 # 预测正确的图片数量 -> Accuracy = correct/total*100%
        total = 0 # 当前epoch已处理的照片总数

        # 遍历训练数据 每次取出一个batch
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播 
            outputs = self.model(inputs)
            # 计算损失 
            loss = self.criterion(outputs, targets)

            # 反向传播 + 更新参数
            optimizer.zero_grad() # 清空旧梯度
            loss.backward() # 计算梯度，并不修改模型参数
            optimizer.step() # 更新参数

            running_loss += loss.item() * inputs.size(0) # loss值*batch大小（数量）
            _, predicted = outputs.max(1) # 最大分数所在的索引
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc


    #评估测试集
    @torch.no_grad()
    def _evaluate(self, testloader):
        self.model.eval()  # 把模型切换到评估模式
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in testloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0) 
            _, predicted = outputs.max(1) 
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    # 控制完整训练过程
    def train_and_evaluate(self, trainloader, testloader, num_epochs, lr):
        optimizer, scheduler = self._build_optimizer(lr)
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        # 进行多个 epoch 
        for epoch in range(num_epochs): #num_epochs=200
            start = time.time()

            '''
            训练完整训练集一次
            在完整测试集上评估一次
            调整学习率'''
            train_loss, train_acc = self._train_one_epoch(trainloader, optimizer)
            test_loss, test_acc = self._evaluate(testloader)
            scheduler.step()

            #记录每次数据
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint(epoch, test_acc)

            # 统计并打印本轮结果
            elapsed = time.time() - start
            print('Epoch [{}/{}] | 用时 {:.1f}s | 当前学习率 {:.5f} | '
                  '训练 loss {:.4f} acc {:.2f}% | 测试 loss {:.4f} acc {:.2f}%'.format(
                      epoch + 1, num_epochs, elapsed, optimizer.param_groups[0]['lr'],
                      train_loss, train_acc, test_loss, test_acc))

        print('训练完成！最佳测试精度: {:.2f}%'.format(best_acc))
        return history, best_acc

    # 把当前训练好的模型状态保存到硬盘，以后可以继续使用
    def _save_checkpoint(self, epoch, acc):
        import os
        os.makedirs('weights', exist_ok=True)
        path = 'weights/{}_best.pth'.format(self.model_name)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'acc': acc}, path)
