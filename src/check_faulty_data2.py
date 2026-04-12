import torch
import torch.nn as nn
import numpy as np
import os

# ===========================
# 설정
# ===========================

DATA_DIR = '/workspace/src/wafer_data'
NORMAL_FILES = [
    "340hz50per.txt", "340hz100per.txt",
    "350hz50per.txt", "350hz100per.txt"
]

FAULTY_FILES = [
    "260hz50per.txt", "260hz70per.txt", "260hz100per.txt",
    "350hz50per.txt", "350hz70per.txt", "350hz100per.txt",
    "240.txt", "280.txt", "340.txt"
]

MODEL_PATH = "/workspace/src/models/autoencoder.pth"
THRESHOLD_K = 3  # k-sigma 기준 (예: 3 시그마)

# ===========================
# 데이터 전처리 함수
# ===========================
def load_and_preprocess_data(file_name):
    data_list = []
    file_path = os.path.join(DATA_DIR, file_name)
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                values = line.strip().split(',')
                if len(values) >= 7:
                    try:
                        selected_data = [float(values[i]) for i in range(1, 7)]
                        data_list.append(selected_data)
                    except ValueError:
                        print(f"잘못된 데이터 형식 무시: {line.strip()}")
    return np.array(data_list, dtype=np.float32)

# =======================
# Autoencoder 모델 정의
# =======================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.01),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.01),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.01),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.01),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.01),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =======================
# 모델 로드
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# =======================
# 임계값 산출
# =======================
def compute_threshold(files, k=3):
    all_errors = []
    for file_name in files:
        data = load_and_preprocess_data(file_name)
        inputs = torch.from_numpy(data).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            errors = torch.mean((inputs - outputs) ** 2, dim=1).cpu().numpy()
            all_errors.extend(errors)
    
    all_errors = np.array(all_errors)
    mean = np.mean(all_errors)
    std = np.std(all_errors)
    threshold = mean + k * std

    print(f"\n[✅ 임계값 설정] 평균: {mean:.2f}, 표준편차: {std:.2f}, 임계값 (k={k}): {threshold:.2f}")
    return threshold

# =======================
# 고장 여부 판별
# =======================
def detect_fault(data, threshold):
    inputs = torch.from_numpy(data).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        reconstruction_error = torch.mean((inputs - outputs) ** 2, dim=1)
    avg_error = reconstruction_error.mean().item()
    fault_ratio = (reconstruction_error > threshold).sum().item() / len(reconstruction_error)
    return fault_ratio, avg_error

# =======================
# 실행
# =======================
THRESHOLD = compute_threshold(NORMAL_FILES, k=THRESHOLD_K)

for file_name in FAULTY_FILES:
    data = load_and_preprocess_data(file_name)
    fault_ratio, avg_error = detect_fault(data, THRESHOLD)

    print(f"\n파일: {file_name}")
    print(f" - 고장 데이터 비율: {fault_ratio * 100:.2f}%")
    print(f" - 평균 재구성 오류: {avg_error:.6f}")
    
    if fault_ratio > 0.5:
        print(f"⚠️ 경고: {file_name}은 고장 데이터입니다!")
    else:
        print(f"✅ 정상: {file_name}은 고장이 아닙니다!")
