# 多模态文本和图像情感分析模型

## 简介
本项目旨在设计一个多模态融合模型，结合文本和图像数据预测情感标签。模型包含文本处理和图像处理分支，并将两者的特征通过一个融合层进行合并，最后输出情感预测。模型使用 ResNet50 作为图像特征提取器，LSTM 处理文本特征，训练集来自 `train.txt`，待预测数据来自 `test_without_label.txt`。

## 执行环境
请确保安装了以下依赖库。这些库都以 pip 包的形式在 requirements.txt 中列出。你可以使用以下命令安装所有依赖库：
```bash
pip install -r requirements.txt
```

## 文件结构
```
.
├── data/                       # 文本和图像数据所在的文件夹
│   ├── guid1.txt               # 文本文件
│   ├── guid1.jpg               # 对应的图像文件
│   ├── guid2.txt
│   ├── guid2.jpg
│   └── ...
├── train.txt                   # 包含训练数据 guid 和 label 的 CSV 文件
├── test_without_label.txt      # 包含待预测数据 guid 和空 label 的 CSV 文件
├── requirements.txt            # 项目依赖库清单
├── multimodal_fusion_model.py  # 包含数据处理、模型构建、训练和预测的所有代码
└── README.md                   # 项目说明文档
```

## 完整流程
### 数据准备
1. 将你的文本和图像数据存放到 `data/` 文件夹，文件名格式为 `guid.txt` 和 `guid.jpg`。
2. 确保 `train.txt` 和 `test_without_label.txt` 文件格式正确，每行两列，第一列 guid，第二列 label，训练集应当只包含有效的 label（positive, neutral, negative）。

### 模型执行
1. 打开命令行终端，确保当前目录为项目根目录。
2. 安装依赖库（如果尚未安装）：`pip install -r requirements.txt`
3. 执行代码：`python multimodal_fusion_model.py`

### 输出结果
预测完成后，`test_without_label.txt` 文件会自动更新，每行的情感标签将被替换为模型预测的结果，并保存到同一文件路径下。

## 参考库
- TensorFlow: 用于机器学习模型的构建和训练。
- pandas: 用于数据的加载和处理。
- numpy: 用于数值计算。
- chardet: 用于自动检测文本文件的编码。

## 代码说明
- `load_data()`: 加载训练数据和测试数据，处理文件路径和标签。
- `preprocess_text()`: 对文本进行分词和填充序列，生成适合模型输入的格式。
- `preprocess_image()`: 对图像进行预处理，包括尺寸调整和 ResNet50 预训练图像预处理。
- `map_labels()`: 将情感标签转换为数值形式。
- `build_multimodal_model()`: 构建一个多模态融合模型，包含图像和文本两个分支。
- `train_model()`: 训练模型并保存训练过程中的相关历史数据。
- `predict()`: 使用训练好的模型预测测试集上的情感标签，并将结果写入到 `test_without_label.txt` 文件中的 label 列。
- 主程序部分：定义各个文件路径、执行数据加载与预处理、模型构建与训练、测试集情感标签预测等。

## 注意事项
- 请确保 `data/` 文件夹中的文本和图像文件编码正确。
- 各个路径需根据项目实际情况进行修改。
- 光运行一次代码可能无法达到满意的效果，需要调整超参数 `epochs` 和 `batch_size` 进行多轮训练和验证。

---

### requirements.txt
```plaintext
tensorflow==2.12.0
pandas==1.4.3
numpy==1.21.5
chardet==4.0.0
```
