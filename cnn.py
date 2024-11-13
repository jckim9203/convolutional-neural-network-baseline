# 필요한 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 데이터 정규화 및 차원 변경 (CNN에 입력할 수 있도록 4차원 텐서로 변환)
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# CNN 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 요약 출력
model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 예측
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 확률이 가장 높은 클래스 선택

# 성능 평가
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

import matplotlib.pyplot as plt

# 테스트 데이터에서 처음 10개 이미지를 시각화하는 코드
plt.figure(figsize=(10, 1))  # 출력할 이미지의 전체 크기 설정
for i in range(10):
    plt.subplot(1, 10, i + 1)  # 1행 10열로 설정하여 각 이미지 위치 지정
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')  # 이미지를 28x28로 변환하여 표시
    plt.axis('off')  # 축 제거
    plt.title(f"Label: {y_test[i]}")
plt.show()

y_pred_classes[0:10]

# 잘못 예측된 인덱스 찾기
wrong_indices = np.where(y_test != y_pred_classes)[0]

# 잘못 예측된 이미지 중 처음 10개를 시각화
plt.figure(figsize=(10, 5))
for i, idx in enumerate(wrong_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
plt.show()
