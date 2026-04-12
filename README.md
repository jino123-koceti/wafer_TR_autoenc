# Wafer TR Autoencoder 고장진단

> 반도체 EFEM(Equipment Front End Module) 웨이퍼 이송 로봇(TR)의 **Z축 풀리벨트 장력 이상**을 오토인코더 기반 비지도 학습으로 감지하는 고장진단 알고리즘

---

## 개요

웨이퍼 이송 로봇의 IMU 센서 데이터를 학습한 오토인코더 모델을 활용하여, **정상 동작 패턴과의 재구성 오차(Reconstruction Error)** 를 고장 판별 기준으로 사용합니다.

- **학습 방식**: 정상 데이터만으로 학습하는 비지도 학습(Unsupervised Learning)
- **고장 판단**: 재구성 오차가 임계값(`56,250,000`)을 초과하면 고장으로 진단
- **진단 대상**: Z축 풀리벨트 장력 이상

---

## 모델 구조

입력 특징: **6D IMU 센서 데이터** (Roll, Pitch, Yaw, Xacc, Yacc, Zacc)

```
[Encoder]  6 → 512 → 256 → 128 → 64 → 32 → 16  (잠재 공간)
[Decoder] 16 →  32 →  64 → 128 → 256 → 512 →  6  (재구성)
```

각 레이어: `Linear → BatchNorm1d → LeakyReLU(0.01)`

---

## 학습 설정

| 항목 | 값 |
|------|----|
| 손실 함수 | L1Loss |
| 옵티마이저 | AdamW (lr=0.005, weight_decay=1e-5) |
| 배치 크기 | 64 |
| 최대 에포크 | 100 |
| 정규화 | Z-score Normalization |
| 그래디언트 클리핑 | max_norm=1.0 |
| Early Stopping | patience=10, min_delta=1e-4 |

**학습 데이터**: 340Hz / 350Hz 정상 동작 데이터 (50%, 100% 속도)

---

## 프로젝트 구조

```
docker_autoenc/
├── Dockerfile                  # CUDA 12.8.1 기반 컨테이너 빌드 설정
├── requirements.txt            # Python 의존성
├── models/
│   └── autoencoder.pth         # 사전 학습된 모델 가중치
└── src/
    ├── train_autoenc.py        # 모델 학습 스크립트
    ├── check_faulty_data.py    # 고장 데이터 검증 스크립트
    ├── check_faulty_data2.py   # 고장 데이터 추가 검증 스크립트
    ├── check_learning_data.py  # 학습 데이터 크기 확인 스크립트
    └── wafer_data/             # 센서 데이터 (로컬 전용, .gitignore 처리)
```

---

## 실행 방법

### 1. Docker 이미지 빌드

```bash
cd ~/ros2_ws/docker_autoenc
sudo docker build -t autoenc-docker .
```

### 2. 모델 학습

```bash
sudo docker run --rm -it \
  -v ~/ros2_ws/docker_autoenc/src/wafer_data:/workspace/src/wafer_data \
  autoenc-docker \
  python /workspace/src/train_autoenc.py
```

### 3. 고장 진단 실행

```bash
sudo docker run --rm -it \
  -v ~/ros2_ws/docker_autoenc/src/wafer_data:/workspace/src/wafer_data \
  -v ~/ros2_ws/docker_autoenc/models:/workspace/src/models \
  autoenc-docker \
  python /workspace/src/check_faulty_data.py
```

### 4. 출력 예시

```
파일: 260-1.txt
 - 평균 재구성 오류: 89432710.234375
⚠️ 진단결과: 고장 데이터입니다! z축 풀리벨트 장력을 확인하십시오.

파일: 350-1.txt
 - 평균 재구성 오류: 12043821.562500
✅ 진단결과: 정상 데이터입니다.
```

---

## 환경 요구사항

- NVIDIA GPU (CUDA 12.8.1 이상)
- Docker
- Python 패키지: `torch`, `torchvision`, `numpy`, `matplotlib`

---

## 데이터 설명

| 파일 패턴 | 설명 |
|-----------|------|
| `{Hz}hz{per}per.txt` | 연속 동작 데이터 (예: `340hz100per.txt` → 340Hz, 100% 속도) |
| `{Hz}-{n}.txt` | 단속 동작 데이터 (예: `260-1.txt` → 260Hz 1회 동작) |

> `src/wafer_data/` 디렉토리는 파일 크기로 인해 저장소에서 제외됩니다 (`.gitignore`).  
> 데이터는 별도로 마운트하여 사용하세요.
