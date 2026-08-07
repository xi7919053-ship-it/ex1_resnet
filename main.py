"""
main.py —— 启动入口
串联 data.py / models / train.py / util.py 完成完整训练流程。

"""

from __future__ import print_function
import argparse

import torch

from data import trainloader, testloader
from models import model_factory
from train import Trainer
from util import plot_history


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.2, type=float, help='Learning Rate')
parser.add_argument('--steps', '-n', default=200, type=int, help='No of Epochs')
parser.add_argument('--gpu', '-p', action='store_true', default=True, help='Train on GPU')
parser.add_argument('--model', '-m', default='resnet18', type=str, help='Name of Network')
args = parser.parse_args()


def main():
    # 检查GPU是否可用（CUDA优先，其次Mac的MPS，都没有就用CPU）
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    train_on_gpu = args.gpu and (has_cuda or has_mps)

    if args.gpu and not train_on_gpu:
        print('警告: 未检测到可用GPU（CUDA/MPS均不可用），将自动切换为CPU训练')
    elif has_cuda:
        print('检测到 NVIDIA GPU，将使用 CUDA 训练')
    elif has_mps:
        print('检测到 Mac Apple Silicon GPU，将使用 MPS 训练')

    # 定义网络结构
    model_name = args.model
    model = model_factory(model_name)

    # 组装 Trainer，串联 3.损失函数 / 4.优化器
    trainer = Trainer(model_name, model, train_on_gpu)

    # 迭代训练 + 评估
    history, best_acc = trainer.train_and_evaluate(
        trainloader, testloader, num_epochs=args.steps, lr=args.lr)

    # 可视化 & 记录最终精度
    plot_history(history)
    print('=' * 50)
    print('最终训练精度记录：')
    print('  最后一个 epoch 训练集精度: {:.2f}%'.format(history['train_acc'][-1]))
    print('  最后一个 epoch 测试集精度: {:.2f}%'.format(history['test_acc'][-1]))
    print('  训练过程中最佳测试集精度: {:.2f}%'.format(best_acc))
    print('=' * 50)


if __name__ == '__main__':
    main()
