
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# ===========================
# 설정
# ===========================

DATA_DIR = '/workspace/src/wafer_data' # Docker 컨테이너 내에서의 경로
NORMAL_FILES = [
    "340hz50per.txt", "340hz100per.txt",
    "350hz50per.txt", "350hz100per.txt"
]

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.005
INPUT_DIM = 6 # 필요한 열: Roll, Pitch, Yaw, Xacc, Yacc, Zacc

#============================
# 데이터 전처리 함수
#============================
def load_and_preprocess_data():
    data_list = []

    for file_name in NORMAL_FILES:
        file_path = os.path.join(DATA_DIR,file_name)

        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():    # 빈줄 무시
                    values = line.strip().split(',')
                    if len(values) >= 7: # 값이 충분한지 확인(필요한 열 개수는 7개)
                        try:
                            # 필요한 열들만 추출(2열 ~ 7열)
                            selected_data = [float(values[i]) for i in range(1,7)]
                            data_list.append(selected_data)
                        except ValueError:
                            print(f"잘못된 데이터 형식 무시: {line.strip()}")
    # 데이터를 numpy 배열로 변환
    data_array = np.array(data_list, dtype=np.float32)
    return data_array

# 데이터 불러오기
data_array = load_and_preprocess_data()
print(f"데이터 로드 완료! 데이터 크기: {data_array.shape}")

# 데이터 정규화
#data_array = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
# Z-score normalization
mean = np.mean(data_array, axis=0)
std = np.std(data_array, axis=0)
data_array = (data_array - mean) / std


# PyTorch DataLoader 준비
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(data_array)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =======================
# Autoencoder 모델 정의
# =======================
class Autoencoder(nn.Module):
    def __init__(self):

        super(Autoencoder, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(6, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16)            
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# =======================
# 모델, 손실 함수, 옵티마이저 설정
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
# criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# ===========================
# Early Stopping 설정
# ===========================
early_stopping_patience = 10  # 개선이 없을 때 멈추는 에폭 수 (예: 10)
min_delta = 1e-4  # 개선으로 인정되는 최소 변화값
best_loss = float('inf')
patience_counter = 0  # 개선되지 않은 에폭 수 카운터



# =======================
# 학습 루프
# =======================

losses = []

# 매 스탭마다 Loss값 출력
''' 
for epoch in range(EPOCHS):
    for batch_idx, data in enumerate(train_loader):
        inputs = data[0].to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
'''
# 매 에포크 마다 Loss값 출력

# 학습 루프
for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()  # 학습 모드 활성화
    for data in train_loader:
        inputs = data[0].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item() * len(inputs)
    
    average_loss = epoch_loss / len(train_loader.dataset)
    losses.append(average_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {average_loss:.4f}")
    
    # ===========================
    # Early Stopping 조건 체크
    # ===========================
    if best_loss - average_loss > min_delta:
        best_loss = average_loss  # 베스트 Loss 갱신
        patience_counter = 0  # 개선되었으므로 카운터 리셋
    else:
        patience_counter += 1  # 개선되지 않았으므로 카운터 증가
    
    # 조기 종료 조건 충족 시 멈춤
    if patience_counter >= early_stopping_patience:
        print(f"\nEarly Stopping Triggered at Epoch {epoch+1}")
        break


# 학습 완료 후 모델 저장하기
model_path = "/workspace/src/autoencoder_test.pth"  # 모델 저장 경로

torch.save(model.state_dict(), model_path)
print(f"Training Complete and Model Saved at {model_path}")

# =======================
# 모델 검증 함수 정의
# =======================
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * len(inputs)
    
    average_loss = total_loss / len(data_loader.dataset)
    return average_loss


# 학습 데이터로 재구성 오류 확인
train_loss = evaluate_model(model, train_loader)
print(f"\nReconstruction Loss on Training Data: {train_loss:.4f}")


# 그래프 저장
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')



plt.savefig('/workspace/src/training_loss.png')
plt.show()