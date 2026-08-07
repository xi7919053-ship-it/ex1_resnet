"""
train.py —— 训练逻辑
对应课件 75-79 页的"训练模型整体过程"五步法：
  1. 准备数据集   (由 data.py 提供，这里直接传入 trainloader/testloader)
  2. 定义网络结构 (由 models 提供，这里传入已实例化的 model)
  3. 定义损失函数 (交叉熵损失，分类任务对应课件 loss function 部分)
  4. 定义优化算法 (SGD + 学习率衰减)
  5. 迭代训练     (前向传播 -> 计算loss -> 反向传播 -> 更新参数)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim


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
        self.model = model.to(self.device)

        # ---- 3. 定义损失函数 ----
        # 分类任务常用交叉熵损失（对应课件 loss function 一节：
        # softmax 输出概率分布 + 交叉熵衡量与真实标签的差距）
        self.criterion = nn.CrossEntropyLoss()

        print('训练设备: {}'.format(self.device))

    def _build_optimizer(self, lr):
        # ---- 4. 定义优化算法 ----
        # 对应课件38-40页：梯度下降法 w <- w - eta * dL/dw
        # 这里用 SGD，并加上 momentum 和 weight_decay 加速收敛、防止过拟合（课件75页超参数设定）
        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=5e-4)
        # 学习率衰减策略：在第100、150个epoch时学习率乘以0.1
        # (常规做法，CIFAR10训练200 epoch时收敛效果较好)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return optimizer, scheduler

    def _train_one_epoch(self, trainloader, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # ---- 5.1 前向传播 ----
            outputs = self.model(inputs)
            # ---- 5.2 计算损失 ----
            loss = self.criterion(outputs, targets)

            # ---- 5.3 反向传播 + 更新参数 ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _evaluate(self, testloader):
        self.model.eval()
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

    def train_and_evaluate(self, trainloader, testloader, num_epochs, lr):
        optimizer, scheduler = self._build_optimizer(lr)
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(num_epochs):
            start = time.time()

            train_loss, train_acc = self._train_one_epoch(trainloader, optimizer)
            test_loss, test_acc = self._evaluate(testloader)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint(epoch, test_acc)

            elapsed = time.time() - start
            print('Epoch [{}/{}] | 用时 {:.1f}s | 当前学习率 {:.5f} | '
                  '训练 loss {:.4f} acc {:.2f}% | 测试 loss {:.4f} acc {:.2f}%'.format(
                      epoch + 1, num_epochs, elapsed, optimizer.param_groups[0]['lr'],
                      train_loss, train_acc, test_loss, test_acc))

        print('训练完成！最佳测试精度: {:.2f}%'.format(best_acc))
        return history, best_acc

    def _save_checkpoint(self, epoch, acc):
        import os
        os.makedirs('weights', exist_ok=True)
        path = 'weights/{}_best.pth'.format(self.model_name)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'acc': acc}, path)
