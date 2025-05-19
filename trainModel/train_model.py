import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Đường dẫn đến dữ liệu đã phân loại theo folder cảm xúc
dataset_path = 'c:/Users/Admin/Downloads/archive/Training/'

# Tạo DataGenerator cho Train và Validation (80/20)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# Xây dựng model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),# chuyền từ 2048 đặc trưng về 128 đạc trưng thông qua 128 neuron. h1= x1.w1+x2.w2+....+ x2048.w2048, cứ thế đến h 128
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

print(f"Số ảnh train: {train_generator.samples}")
print(f"Số batch train mỗi epoch: {len(train_generator)}")
print(f"Số ảnh validation: {validation_generator.samples}")
print(f"Số batch validation mỗi epoch: {len(validation_generator)}")