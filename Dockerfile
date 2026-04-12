# 베이스 이미지 (Ubuntu 24.04 기반 CUDA 12.8.1)
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# 작업 디렉토리 설정
WORKDIR /workspace

# 가상환경 생성 및 활성화
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# 종속성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY src/check_learning_data.py ./src/check_learning_data.py  

# 작업 디렉토리 변경
WORKDIR /workspace/src

# 실행 명령어 설정
CMD ["python3", "train_autoenc.py"]
