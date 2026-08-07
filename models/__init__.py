"""
models/__init__.py 
整理这个models文件夹对外提供哪些功能
把models文件夹变成一个方便导入和使用的Python包
"""

from .resnet import ResNet18
# from models.resnet import ResNet18


# 模型工厂：接收模型名称，创建对应模型
def model_factory(model_name, num_classes=10):
    # model_factory('resnet18', 10)
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError('Invalid model name: {}，only resnet18 is available'.format(model_name))
