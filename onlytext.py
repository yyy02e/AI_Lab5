import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd

# 配置设备
device = torch.device("cpu")

# 自定义 collate_fn 处理变长输入
def collate_fn(batch):
    """
    自定义的 collate_fn 用于处理变长的文本输入
    """
    # 获取文本输入和标签
    text_inputs, labels = zip(*batch)

    # 获取所有文本输入的 input_ids 和 attention_mask
    input_ids = [text_input['input_ids'].squeeze(0) for text_input in text_inputs]
    attention_mask = [text_input['attention_mask'].squeeze(0) for text_input in text_inputs]

    # 确定最大长度
    max_length = max([input_id.size(0) for input_id in input_ids])

    # 对文本输入做 padding，确保所有文本输入的长度一致
    for i in range(len(input_ids)):
        padding_length = max_length - input_ids[i].size(0)
        input_ids[i] = torch.cat([input_ids[i], torch.zeros(padding_length, dtype=torch.long)], dim=0)
        attention_mask[i] = torch.cat([attention_mask[i], torch.zeros(padding_length, dtype=torch.long)], dim=0)

    # 对 input_ids 和 attention_mask 做堆叠
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)

    # 转换标签为 Tensor
    labels = torch.tensor(labels)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

# 仅文本数据的加载类
class TextOnlyDataset(Dataset):
    def __init__(self, data_folder, label_file, tokenizer, max_length=512):
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载标签文件
        self.data = pd.read_csv(label_file)
        self.guid_list = self.data["guid"].tolist()  # 获取所有的guid
        self.labels = self.data["tag"].map({"positive": 0, "neutral": 1, "negative": 2}).tolist()  # 标签映射为数字

    def __len__(self):
        return len(self.guid_list)

    def __getitem__(self, idx):
        guid = self.guid_list[idx]
        text_path = os.path.join(self.data_folder, f"{guid}.txt")

        # 读取文本
        with open(text_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()

        # 使用BERT分词器处理文本
        text_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_length)

        # 返回文本输入和标签
        label = self.labels[idx]
        return text_input, label

# 仅文本模型（BERT）
class TextOnlyModel(nn.Module):
    def __init__(self):
        super(TextOnlyModel, self).__init__()
        self.bert = BertModel.from_pretrained('/Users/cynthia/PycharmProjects/AI_Lab5/bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 3)  # 输出3个类别：positive, neutral, negative

    def forward(self, text_input):
        text_output = self.bert(**text_input)
        output = self.fc(text_output.pooler_output)
        return output

def train_model(model, train_loader, epochs=3, learning_rate=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, labels in train_loader:
            optimizer.zero_grad()
            data = {key: val.squeeze(1).to(device) for key, val in data.items()}

            labels = labels.to(device)

            # 前向传播
            outputs = model(data)
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
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

# 设置 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('/Users/cynthia/PycharmProjects/AI_Lab5/bert-base-uncased')

# 加载仅文本数据集
train_dataset = TextOnlyDataset(data_folder='/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/data', label_file='/Users/cynthia/PycharmProjects/AI_Lab5/实验五数据/train.txt', tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 初始化模型
model = TextOnlyModel().to(device)

# 训练模型
train_model(model, train_loader)
