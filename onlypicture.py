import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
from PIL import Image

# 配置设备
device = torch.device("cpu")


# 定义图像数据加载类
class ImageOnlyDataset(Dataset):
    def __init__(self, data_folder, label_file, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # 加载标签文件
        self.data = pd.read_csv(label_file)
        self.guid_list = self.data["guid"].tolist()  # 获取所有的guid
        self.labels = self.data["tag"].map({"positive": 0, "neutral": 1, "negative": 2}).tolist()  # 标签映射为数字

    def __len__(self):
        return len(self.guid_list)

    def __getitem__(self, idx):
        guid = self.guid_list[idx]
        image_path = os.path.join(self.data_folder, f"{guid}.jpg")

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 返回图像数据和标签
        label = self.labels[idx]
        return image, label


# 定义图像模型（仅使用ResNet50）
class ImageOnlyModel(nn.Module):
    def __init__(self, pretrained_weights_path=None):
        super(ImageOnlyModel, self).__init__()
        # 加载ResNet50模型
        self.resnet = models.resnet50(pretrained=False)  # 不加载默认的预训练权重
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 3)  # 输出3个类别：positive, neutral, negative

        # 如果提供了本地预训练权重，则加载
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, path):
        """
        加载本地的ResNet50预训练权重
        """
        print(f"Loading pretrained weights from {path}...")

        # 获取预训练权重的state_dict
        pretrained_dict = torch.load(path)

        # 移除与当前模型不匹配的层（例如fc层）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}

        # 加载适配后的权重
        self.resnet.load_state_dict(pretrained_dict, strict=False)

    def forward(self, image):
        image_features = self.resnet(image)
        return image_features


# 设置图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据路径和标签文件
data_folder = '/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/data'  # 替换为你的数据路径
label_file = '/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/train.txt'  # 替换为你的标签文件路径

# 加载图像数据集
train_dataset = ImageOnlyDataset(data_folder, label_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 本地预训练模型权重路径
pretrained_weights_path = '/Users/cynthia/PycharmProjects/AI_Lab5/resnet50-0676ba61.pth'  # 替换为你本地保存的权重路径

# 初始化模型
model = ImageOnlyModel(pretrained_weights_path=pretrained_weights_path).to(device)

# 训练模型
def train_model(model, train_loader, epochs=3, learning_rate=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            # 将数据移动到设备（GPU 或 CPU）
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # 保存模型权重
        torch.save(model.state_dict(), f"image_only_model_epoch_{epoch + 1}.pth")


# 调用训练函数
train_model(model, train_loader)
