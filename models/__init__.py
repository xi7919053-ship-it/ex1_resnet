"""
models/__init__.py —— 模型工厂
提供 model_factory(name) 函数，按名称返回对应的模型实例，
方便 main.py 通过命令行参数 --model 切换不同网络（对照课件80页项目结构）。
"""

from .resnet import ResNet18


def model_factory(model_name, num_classes=10):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError('未知的模型名称: {}，目前仅实现了 resnet18'.format(model_name))
