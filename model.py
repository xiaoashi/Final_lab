import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.applications import ResNet50
from tf_keras.layers import Dense, Input, GlobalAveragePooling2D, Concatenate, Embedding, LSTM, Dropout
from tf_keras.models import Model
from tf_keras.optimizers import Adam
import chardet

# --------------------
# GPU配置 (可选)
# --------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --------------------
# 数据加载与预处理
# --------------------
def load_data(train_path, test_path, data_dir):
    # 加载 train 数据
    train_data = pd.read_csv(train_path, sep=',', header=None, names=['guid', 'label'])
    train_data = train_data.iloc[1:]  # 去除表头
    train_data = train_data[train_data['label'].isin(['positive', 'neutral', 'negative'])]  # 过滤无效标签
    train_texts = []
    for guid in train_data['guid']:
        file_path = os.path.join(data_dir, f"{guid}.txt")
        try:
            with open(file_path, 'rb') as f:
                detected_encoding = chardet.detect(f.read())['encoding']
            with open(file_path, encoding=detected_encoding, errors='ignore') as f:
                train_texts.append(f.read())
        except Exception as e:
            print(f"Error reading file: {file_path}, Error: {e}")
    train_images = [os.path.join(data_dir, f"{guid}.jpg") for guid in train_data['guid']]
    train_labels = train_data['label']

    # 加载 test 数据
    test_data = pd.read_csv(test_path, sep=',', header=None, names=['guid', 'label'])
    test_data = test_data.iloc[1:]  # 去除表头
    test_texts = []
    for guid in test_data['guid']:
        file_path = os.path.join(data_dir, f"{guid}.txt")
        try:
            with open(file_path, 'rb') as f:
                detected_encoding = chardet.detect(f.read())['encoding']
            with open(file_path, encoding=detected_encoding, errors='ignore') as f:
                test_texts.append(f.read())
        except Exception as e:
            print(f"Error reading file: {file_path}, Error: {e}")
    test_images = [os.path.join(data_dir, f"{guid}.jpg") for guid in test_data['guid']]

    print(f"Loaded {len(train_texts)} training texts, {len(train_images)} training images.")
    return train_texts, train_images, train_labels, test_texts, test_images


def preprocess_text(texts, max_len=128, vocab_size=10000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    print(f"Processed text shape: {padded_sequences.shape}")
    return padded_sequences, tokenizer


def preprocess_image(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        image = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        images.append(image)
    images = np.array(images)
    print(f"Processed image shape: {images.shape}")
    return images


# --------------------
# 标签映射与转换
# --------------------
def map_labels(labels, label_mapping):
    labels = labels.astype(str)
    mapped_labels = np.array([label_mapping[label] for label in labels if label in label_mapping])
    return mapped_labels


# --------------------
# 模型设计
# --------------------
def build_multimodal_model(text_vocab_size, text_max_len, num_classes=3):
    # 文本分支
    text_input = Input(shape=(text_max_len,), name="Text_Input")
    embedding = Embedding(input_dim=text_vocab_size, output_dim=128)(text_input)
    lstm = LSTM(128)(embedding)
    text_output = Dropout(0.5)(lstm)

    # 图像分支
    image_input = Input(shape=(224, 224, 3), name="Image_Input")
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    image_output = GlobalAveragePooling2D()(base_model.output)

    # 融合层
    combined = Concatenate()([text_output, image_output])
    dense = Dense(128, activation='relu')(combined)
    dropout = Dropout(0.5)(dense)
    output = Dense(num_classes, activation='softmax')(dropout)

    # 模型
    model = Model(inputs=[text_input, image_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 输出模型摘要信息
    model.summary()
    return model


# --------------------
# 模型训练与验证
# --------------------
def train_model(model, train_texts, train_images, train_labels, val_texts, val_images, val_labels, epochs=5,
                batch_size=8):
    print(f"Training data text shape: {train_texts.shape}")
    print(f"Training data image shape: {train_images.shape}")
    print(f"Validation data text shape: {val_texts.shape}")
    print(f"Validation data image shape: {val_images.shape}")

    history = model.fit(
        [train_texts, train_images], train_labels,
        validation_data=([val_texts, val_images], val_labels),
        epochs=epochs, batch_size=batch_size
    )
    return history


# --------------------
# 测试集预测
# --------------------
# --------------------
# 测试集预测并直接更新原文件
# --------------------
def predict(model, test_texts, test_images, test_file_path, output_path):
    # 加载测试文件
    test_data = pd.read_csv(test_file_path, sep=',', header=None, names=['guid', 'label'])

    # 进行模型预测
    predictions = model.predict([test_texts, test_images])
    predicted_labels = np.argmax(predictions, axis=1)

    # 标签映射
    reverse_label_mapping = {0: 'positive', 1: 'neutral', 2: 'negative'}

    # 替换空值（null 或 NaN）为预测标签
    test_data['label'] = test_data['label'].fillna(
        pd.Series(predicted_labels).map(reverse_label_mapping)
    )

    # 保存修改后的文件
    test_data.to_csv(test_file_path, index=False, header=False)
    print(f"Updated test file saved to {test_file_path}")

# --------------------
# 主程序
# --------------------
if __name__ == "__main__":
    train_path = "train.txt"
    test_path = "test_without_label.txt"
    data_dir = "data"
    output_path = "test_predictions.txt"

    # 加载数据
    train_texts, train_images, train_labels, test_texts, test_images = load_data(train_path, test_path, data_dir)

    # 标签映射
    label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
    train_labels = train_labels.astype(str)
    train_labels = map_labels(train_labels, label_mapping)

    # 划分训练集和验证集
    train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(
        train_texts, train_images, train_labels, test_size=0.2, random_state=42
    )

    # 检查验证集标签
    if np.issubdtype(val_labels.dtype, np.number):
        print("Validation labels are already numeric, no need to map.")
    else:
        val_labels = val_labels.astype(str)  # 确保为字符串格式
        val_labels = map_labels(val_labels, label_mapping)

    # 文本预处理，保存 tokenizer
    train_texts, tokenizer = preprocess_text(train_texts)
    val_texts, _ = preprocess_text(val_texts, vocab_size=len(tokenizer.word_index) + 1)
    test_texts, _ = preprocess_text(test_texts, vocab_size=len(tokenizer.word_index) + 1)

    # 图像预处理
    train_images = preprocess_image(train_images)
    val_images = preprocess_image(val_images)
    test_images = preprocess_image(test_images)

    # 构建模型
    model = build_multimodal_model(text_vocab_size=len(tokenizer.word_index) + 1, text_max_len=train_texts.shape[1])

    # 训练模型
    train_model(model, train_texts, train_images, train_labels, val_texts, val_images, val_labels)

    # 预测测试集
    predict(model, test_texts, test_images, test_path, output_path)

