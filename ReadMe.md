## 项目概述
这是一个多模态情感分析项目，结合图像和文本数据，使用深度学习模型进行情感分类。图像特征通过 ResNet50 提取，文本特征通过 BERT 提取，最后融合这两者进行情感分类。情感标签包括 **positive**、**neutral** 和 **negative**。

## 项目结构
```
/MultiModal-Sentiment-Analysis
│
├── 实验五数据/                        # 数据文件夹，包含图像和文本文件
│   ├── train.txt                # 训练数据标签文件
│   ├── test_without_label.txt   # 测试数据文件（没有标签）
│   ├── data                     # 图像文件以及文本文件
│   
├── ├── resnet50-0676ba61.pth    # ResNet50 模型权重
│   └── bert-base-uncased/       # BERT 模型文件
│   └── model.pth.               # 训练好后的模型权重
├── main.py                      # 主程序，包含模型训练、验证和测试逻辑
├── onlytext.py                   #消融实验只输入文本的训练验证
├── onlypicture.py                #消融实验只输入图像数据的训练验证
├── result.txt                    #test预测结果
├── requirements.txt             # 所需依赖库
├── README.md                    # 项目的说明文档
```

## 安装与环境配置

### 安装依赖
1. 创建虚拟环境并激活：

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. 安装所需的依赖：

   ```bash
   pip install -r requirements.txt
   ```

### 依赖库

`requirements.txt` 文件中列出了以下依赖库：

```
torch==1.10.0
transformers==4.11.3
torchvision==0.11.1
scikit-learn==0.24.2
pandas==1.3.3
opencv-python==4.5.3.56
Pillow==8.4.0
```

## 使用方法

1. **数据准备**：
   - 将训练数据（图像和文本）放置在 `data` 文件夹中。
   - `train.txt` 文件包含图像和其对应的情感标签。
   - `test_without_label.txt` 文件用于存放待预测的测试数据。

2. **训练模型**：
   使用以下命令开始训练模型：

   ```bash
   python main.py
   ```

3. **预测结果**：
   - 训练结束后，模型会在 `test_without_label.txt` 上进行预测。
   - 预测结果会被保存到 `result.txt` 文件中，格式为：

   ```
   guid,tag
   1,positive
   2,negative
   3,neutral
   ```

## 项目功能

- 使用 **ResNet50** 提取图像特征。
- 使用 **BERT** 提取文本特征。
- 融合图像和文本特征并进行情感分类。
- 提供训练、验证和测试的流程。
- 输出情感分类结果。
