
#============== 독커 빌드 ==========================#
#cd ~/ros2_ws/docker_autoenc
#sudo docker build -t autoenc-docker .


#============== model 마운트 해서 독커 실행 ==========#
#sudo docker run --rm -it \
#  -v ~/ros2_ws/docker_autoenc/models:/workspace/src/models \
#  autoenc-docker \
#  python /workspace/src/check_faulty_data.py


#============== 데이터 추가 ========================#
#sudo docker run --rm -it \
#  -v ~/ros2_ws/docker_autoenc/src/wafer_data:/workspace/src/wafer_data \
#  -v ~/ros2_ws/docker_autoenc/models:/workspace/src/models \
#  autoenc-docker \
#  python /workspace/src/check_faulty_data.py


import torch
import torch.nn as nn
import numpy as np
import os

# ===========================
# 설정
# ===========================

DATA_DIR = '/workspace/src/wafer_data'  # Docker 컨테이너 내 경로
FAULTY_FILES = [
#    "260hz50per.txt", "260hz70per.txt", "260hz100per.txt",
#    "280hz50per.txt", "280hz70per.txt", "280hz100per.txt",
#    "300hz50per.txt", "300hz70per.txt", "300hz100per.txt", 
#    "320hz50per.txt", "320hz70per.txt", "320hz100per.txt", 
#    "330hz50per.txt", "330hz70per.txt", "330hz100per.txt", 
#    "340hz50per.txt", "340hz70per.txt", "340hz100per.txt", 
#    "350hz50per.txt", "350hz70per.txt", "350hz100per.txt" ,
#    "240.txt", "280.txt", "340.txt" , 
#    "1-1.txt", "1-2.txt", "1-3.txt", "1-4.txt", "1-5.txt", 
#    "2-1.txt", "2-2.txt", "2-3.txt", "2-4.txt", "2-5.txt",
#    "3-1.txt", "3-2.txt", "3-3.txt", "3-4.txt", "3-5.txt",   
    "260-1.txt", "260-2.txt", "260-3.txt", "260-4.txt", 
    "260-5.txt", "260-6.txt", "260-7.txt", "260-8.txt", "260-9.txt",
    "260-10.txt",
    "350-1.txt", "350-2.txt", "350-3.txt", "350-4.txt", 
    "350-5.txt", "350-6.txt", "350-7.txt", "350-8.txt", "350-9.txt",
    "350-10.txt",
# #    "360hz100per.txt", "360hz100per2.txt", "360hz100per3.txt", 
##    "300hz100per2.txt", "300hz100per3.txt", "300hz100per4.txt", 
##    "260hz100per2.txt", "260hz100per3.txt" 
    ]

MODEL_PATH = "/workspace/src/models/autoencoder.pth"
THRESHOLD = 56250000  # 임계값 (재구성 오류가 이 값을 넘으면 고장으로 판단)

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
    data_array = np.array(data_list, dtype=np.float32)
    return data_array

# =======================
# Autoencoder 모델 정의
# =======================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
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
# 모델 로드
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# =======================
# 고장 데이터 검출
# =======================
def detect_fault(data, threshold):
    inputs = torch.from_numpy(data).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        reconstruction_error = torch.mean((inputs - outputs) ** 2, dim=1)
        
    # 평균 재구성 오류 계산
    average_reconstruction_error = torch.mean(reconstruction_error).item()
    
    # 고장 데이터 판별
    fault_count = torch.sum(reconstruction_error > threshold).item()
    fault_ratio = fault_count / len(reconstruction_error)
    
    return fault_ratio, average_reconstruction_error
# =======================
# 고장 데이터 판별
# =======================
for file_name in FAULTY_FILES:
    data = load_and_preprocess_data(file_name)
    fault_ratio, avg_error = detect_fault(data, THRESHOLD)

    is_fault = avg_error > THRESHOLD


# =======================
# 고장 데이터 판별
# =======================
for file_name in FAULTY_FILES:
    data = load_and_preprocess_data(file_name)
    fault_ratio, avg_error = detect_fault(data, THRESHOLD)

    # 파일명으로 기대 상태 결정
    is_fault = avg_error > THRESHOLD       # ✅ 평균 오류 기준으로 고장 판별

    print(f"\n파일: {file_name}")
#    print(f" - 고장 데이터 비율: {fault_ratio * 100:.2f}%")
    print(f" - 평균 재구성 오류: {avg_error:.6f}")

    if is_fault:
        print(f"⚠️ 진단결과: 고장 데이터입니다! z축 풀리벨트 장력을 확인하십시오.")
    else:
        print(f"✅ 진단결과: 정상 데이터입니다.")

#    if is_fault != expected_faulty:
#        print(f"❗ [주의] 예측이 기대값과 다릅니다. 파일: {file_name}")
