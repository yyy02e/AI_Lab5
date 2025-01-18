import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import pandas as pd  # 用于加载train.txt和test_without_label.txt
# 自定义 collate_fn 处理变长输入
import torch
def collate_fn(batch):
    """
    自定义的 collate_fn 用于处理变长的文本输入和图像
    """
    # 获取图像、文本输入和标签
    images, text_inputs, labels = zip(*batch)

    # 使用 torch 对图像进行堆叠
    images = torch.stack(images, 0)

    # 手动处理文本输入的 padding
    input_ids = [text_input['input_ids'].squeeze(0) for text_input in text_inputs]
    attention_mask = [text_input['attention_mask'].squeeze(0) for text_input in text_inputs]

    # 处理 padding
    max_length = 512  # 设置最大长度，确保所有文本填充到相同长度
    for i in range(len(input_ids)):
        # 如果长度小于最大长度，则填充
        if input_ids[i].size(0) < max_length:
            padding_length = max_length - input_ids[i].size(0)
            input_ids[i] = torch.cat([input_ids[i], torch.zeros(padding_length, dtype=torch.long)], dim=0)
            attention_mask[i] = torch.cat([attention_mask[i], torch.zeros(padding_length, dtype=torch.long)], dim=0)

    # 对文本输入做 padding，确保所有文本输入的长度一致
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)

    # 转换标签为 Tensor
    labels = torch.tensor(labels)

    return images, {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

device = torch.device("cpu")  # 如果你想使用GPU，替换为 "cuda"

# 数据加载类：用于加载图像和文本数据
class MultiModalDataset(Dataset):
    def __init__(self, data_folder, label_file, tokenizer, transform=None):
        """
        初始化数据集
        :param data_folder: 存放图像和文本文件的文件夹路径
        :param label_file: 存放标签的文件路径（train.txt）
        :param tokenizer: 用于文本处理的BERT分词器
        :param transform: 图像处理（预处理）函数
        """
        self.data_folder = data_folder  # 图像和文本文件夹路径
        self.tokenizer = tokenizer  # 用于处理文本的BERT分词器
        self.transform = transform  # 图像预处理

        # 加载标签文件（train.txt），假设文件中有guid和label两列
        self.data = pd.read_csv(label_file)  # 加载CSV文件，不再需要delimiter=" "

        # 图像和文本的对应
        self.guid_list = self.data["guid"].tolist()  # 获取所有的guid
        self.labels = self.data["tag"].map({"positive": 0, "neutral": 1, "negative": 2}).tolist()  # 标签转换为数字

    def __len__(self):
        return len(self.guid_list)

    def __getitem__(self, idx):
        """
        获取一个样本的数据（图像和文本）
        :param idx: 索引
        :return: 图像、文本和标签
        """
        guid = self.guid_list[idx]  # 获取当前数据的guid
        image_path = os.path.join(self.data_folder, f"{guid}.jpg")  # 图像路径
        text_path = os.path.join(self.data_folder, f"{guid}.txt")  # 文本路径

        # 加载图像
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)  # 应用图像预处理

        with open(text_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()

        # 使用BERT分词器处理文本
        text_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 返回图像、文本输入和标签
        label = self.labels[idx]
        return image, text_input, label


# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用本地路径加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('/Users/cynthia/PycharmProjects/AI_Lab5/bert-base-uncased')

# 创建数据集实例
data_folder = '/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/data'  # 图像和文本都在data文件夹下
label_file = '/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/train.txt'  # 训练集标签文件路径
dataset = MultiModalDataset(data_folder, label_file, tokenizer, transform)

# 划分训练集和验证集
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)


# 训练过程中的模型定义
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()

        # 图像处理部分：使用本地加载的ResNet50模型
        resnet_path = '/Users/cynthia/PycharmProjects/AI_Lab5/resnet50-0676ba61.pth'  # 本地保存的ResNet50权重路径
        self.resnet = models.resnet50()
        self.resnet.load_state_dict(torch.load(resnet_path, weights_only=True))  # 加载本地预训练的ResNet50权重
        self.resnet.fc = nn.Identity()  # 去掉ResNet的全连接层

        # 文本处理部分：使用本地加载的BERT模型
        self.bert = BertModel.from_pretrained('/Users/cynthia/PycharmProjects/AI_Lab5/bert-base-uncased')
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, 768)  # 将BERT的输出映射到768维

        # 融合部分：连接图像和文本特征
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)  # 输出3个类：positive, neutral, negative
        )

    def forward(self, image, text_input):
        # 图像特征
        image_features = self.resnet(image)

        # 文本特征
        text_output = self.bert(**text_input)
        text_features = self.bert_fc(text_output.pooler_output)

        # 融合图像和文本特征
        combined_features = torch.cat((image_features, text_features), dim=1)

        # 通过全连接层进行分类
        output = self.fc(combined_features)
        return output


# 训练模型的函数
def train_model(model, train_loader, val_loader, epochs=5, learning_rate=1e-5, save_path="model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, text_inputs, labels in train_loader:
            optimizer.zero_grad()

            # 移动图像和标签到指定的设备
            images, labels = images.to(device), labels.to(device)

            # 分别将 BERT 的输入移到设备
            text_inputs['input_ids'] = text_inputs['input_ids'].to(device)
            text_inputs['attention_mask'] = text_inputs['attention_mask'].to(device)

            # 前向传播
            outputs = model(images, text_inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # 每个epoch保存模型
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # 你可以在此添加验证集评估代码，查看模型在验证集上的表现
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, text_inputs, labels in val_loader:
                # 移动图像和标签到指定的设备
                images, labels = images.to(device), labels.to(device)

                # 分别将 BERT 的输入移到设备
                text_inputs['input_ids'] = text_inputs['input_ids'].to(device)
                text_inputs['attention_mask'] = text_inputs['attention_mask'].to(device)

                outputs = model(images, text_inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")
        print(f"Validation Accuracy: {correct / total * 100:.2f}%")

# 测试模型的函数
def test_model(model, test_loader, model_path="model.pth"):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, text_inputs, _ in test_loader:
            # 移动图像和 BERT 的输入到设备
            images = images.to(device)
            text_inputs['input_ids'] = text_inputs['input_ids'].to(device)
            text_inputs['attention_mask'] = text_inputs['attention_mask'].to(device)

            outputs = model(images, text_inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    return predictions

def generate_test_results(test_df, predictions, output_file="result.txt"):
    predictions_length = len(predictions)
    test_length = len(test_df)

    # 处理长度不匹配的情况
    if predictions_length < test_length:
        predictions.extend([predictions[-1]] * (test_length - predictions_length))  # 填充最后一个预测值
    elif predictions_length > test_length:
        predictions = predictions[:test_length]

    # 将预测结果添加到 DataFrame 中
    test_df['tag'] = [map_label(p) for p in predictions]  # 使用预测值填充标签列

    # 检查文件是否存在，如果存在就避免重复写入列头
    if os.path.exists(output_file):
        # 如果文件已存在，设置 header=False 避免写入列标题
        test_df[['guid', 'tag']].to_csv(output_file, index=False, header=False, mode='a', sep=",")
        print(f"Appended results to {output_file}")
    else:
        # 如果文件不存在，设置 header=True 写入列标题
        test_df[['guid', 'tag']].to_csv(output_file, index=False, header=True, sep=",")
        print(f"Created {output_file} and added results")

# 标签映射函数
def map_label(prediction):
    # 映射预测标签到字符串
    if prediction == 0:
        return "positive"
    elif prediction == 1:
        return "neutral"
    else:
        return "negative"


# 主程序：训练和测试
if __name__ == "__main__":
    # 加载数据
    test_file = '/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/test_without_label.txt'
    # 使用 header=None，确保 pandas 不把第一行当作列标题
    test_df = pd.read_csv(test_file, header=None, names=["guid", "tag"], skipinitialspace=True)

    # 创建测试集数据加载器
    test_dataset = MultiModalDataset(data_folder, test_file, tokenizer, transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # 使用 `to(device)` 将模型和数据移至 CPU 或 GPU
    model = MultiModalModel().to(device)

    # 训练模型
    train_model(model, train_loader, val_loader, epochs=5, learning_rate=1e-5)

    # 加载训练好的模型
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    # 测试模型并生成结果
    predictions = test_model(model, test_loader, model_path="model.pth")

    # 保存结果到 result.txt 文件
    generate_test_results(test_df, predictions, output_file="result.txt")
