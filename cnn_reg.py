# 필요한 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 향상된 CNN 모델 구축
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

# 모델 요약 출력
model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 예측
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 성능 평가
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

import matplotlib.pyplot as plt

# 학습 데이터에서 처음 10개 이미지를 시각화하는 코드
plt.figure(figsize=(10, 1))  # 출력할 이미지의 전체 크기 설정
for i in range(10):
    plt.subplot(1, 10, i + 1)  # 1행 10열로 설정하여 각 이미지 위치 지정
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')  # 이미지를 28x28로 변환하여 표시
    plt.axis('off')  # 축 제거
    plt.title(f"Label: {y_test[i]}")
plt.show()
