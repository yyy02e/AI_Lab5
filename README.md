# AI_Lab5
## 项目描述
本项目实现了一个多模态情感分析模型，结合图像和文本数据进行情感分类。模型使用了 ResNet50 用于图像特征提取，BERT 用于文本特征提取，最终通过一个全连接层融合图像和文本特征进行情感分类。情感标签包括正面（positive）、中立（neutral）和负面（negative）。

## 技术栈
- **PyTorch**：用于深度学习模型的实现。
- **Transformers**：用于加载和使用 BERT 模型。
- **OpenCV**：用于图像处理和加载。
- **scikit-learn**：用于数据分割和评估。
- **pandas**：用于数据处理和加载 CSV 文件。
- **torchvision**：用于加载和使用预训练的 ResNet50 模型。

## 文件结构
```
/AI_Lab5
│
├── 实验五数据/                        # 数据文件夹，包含图像和文本文件
│   ├── train.txt                # 训练数据标签文件
│   ├── test_without_label.txt   # 测试数据文件（没有标签）
│   ├── data                     # 图像文件以及文本数据
│  
│
├── ├── resnet50-0676ba61.pth     # ResNet50 模型权重
│   └── bert-base-uncased/        # BERT 模型文件           
│
├── main.py                      # 主程序，包含模型训练、验证和测试逻辑
├── requirements.txt             # 所需依赖库
├── README.md                    # 项目的说明文档
```

## 环境配置
要运行本项目，请先确保你安装了以下依赖项。可以通过以下命令安装所有依赖：

1. 克隆该仓库到本地：
   ```bash
   git clone https://github.com/yyy02e/AI_Lab5.git
   cd AI_Lab5
   ```

2. 创建一个虚拟环境并激活：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate     # For Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 依赖库
将以下内容保存到 `requirements.txt` 文件中：

```
torch==1.10.0
transformers==4.11.3
torchvision==0.11.1
scikit-learn==0.24.2
pandas==1.3.3
opencv-python==4.5.3.56
Pillow==8.4.0
```

## 执行代码的完整流程

1. **数据准备**：
   将图像和文本数据放置到 `实验五数据` 文件夹中。`train.txt` 文件包含图像对应的情感标签，而 `test_without_label.txt` 包含待预测的图像数据。

2. **训练模型**：
   运行以下命令开始训练模型：

   ```bash
   python main.py
   ```

   在训练过程中，模型将使用 ResNet50 和 BERT 分别处理图像和文本数据，最终进行多模态融合，输出情感分类结果。

3. **验证和测试**：
   训练结束后，模型会自动进行验证并保存训练好的模型权重。你可以使用训练好的模型在测试集上进行预测。

   预测结果将被保存到 `result.txt` 文件中，格式为：
   ```
   guid,tag
   1,positive
   2,negative
   3,neutral
   ```

## 参考的库和实现
1. **PyTorch**: 用于模型的训练和推理。
   - [PyTorch Official Site](https://pytorch.org/)
   
2. **Transformers by Hugging Face**: 用于加载和处理 BERT 模型。
   - [Transformers Documentation](https://huggingface.co/transformers/)
   
3. **ResNet50**: 预训练的图像分类模型，用于提取图像特征。
   - [ResNet50 in torchvision](https://pytorch.org/vision/stable/models.html#resnet)

4. **其他参考资料**:
   - [A Deep Dive into Multi-Modal Learning](https://arxiv.org/abs/1906.04794)
   - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 常见问题

### 1. **如何使用 GPU 进行训练？**
   - 如果你有 GPU 并且 CUDA 已正确安装，修改代码中的 `device = torch.device("cuda")` 来启用 GPU。
   - 确保你的 PyTorch 安装是支持 CUDA 的版本。

### 2. **如何更改模型的超参数？**
   - 你可以在 `train_model` 函数中调整 `epochs`、`learning_rate` 和 `batch_size` 等超参数。

### 3. **如何调整数据预处理步骤？**
   - 你可以在 `transform` 变量中调整图像的大小、归一化参数等设置，或者修改 BERT 分词器的参数（如 `max_length`）。

## 联系方式
- GitHub: [https://github.com/yyy02e/AI_Lab5](https://github.com/yyy02e/AI_Lab5)
- Email: 13611872105@163.com
