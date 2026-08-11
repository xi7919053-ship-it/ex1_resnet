import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.resnet import ResNet18



# CIFAR-10 10个类别
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# 图片预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])


# 选择设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("使用设备:", device)


# 创建模型
model = ResNet18(num_classes=10).to(device)

# 加载训练好的模型参数
checkpoint = torch.load(
    "weights/resnet18_best.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])


# 切换到评估模式
model.eval()


# 遍历 images 文件夹
image_dir = "images"

for filename in os.listdir(image_dir):

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, filename)

    # 打开图片并转换为RGB
    image = Image.open(image_path).convert("RGB")

    # 预处理
    image_tensor = transform(image)

    # 增加Batch维度
    # [3,32,32] -> [1,3,32,32]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 推理阶段
    # ！！！注意注意此处不需要计算梯度
    with torch.no_grad():
        outputs = model(image_tensor)

        # 将logits转换为概率
        probabilities = F.softmax(outputs, dim=1)

        # 找到概率最大的类别 -> 即为预测结果
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_value = confidence.item() * 100

    print(
        f"{filename:15s} -> "
        f"{predicted_class:10s} "
        f"({confidence_value:.2f}%)"
    )

# 显示图片
plt.imshow(image)
plt.axis("off")

# 设置标题
plt.title(f"Prediction: {predicted_class} ({confidence_value:.2f}%)")

plt.show()
