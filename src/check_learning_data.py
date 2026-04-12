import os
import numpy as np

# 데이터 경로 설정 (수정됨)
DATA_DIR = './wafer_data'

# 사용할 데이터 파일 목록
DATA_FILES = [
    '340hz50per.txt', '340hz70per.txt', '340hz100per.txt',
    '350hz50per.txt', '350hz70per.txt', '350hz100per.txt'
]

# 각 파일별 데이터 크기를 저장할 딕셔너리
data_sizes = {}

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data():
    data_list = []

    for file_name in DATA_FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        
        file_data_list = []  # 개별 파일의 데이터 저장
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # 빈 줄 무시
                    values = line.strip().split(',')
                    if len(values) >= 7:  # 필요한 열 개수는 7개
                        try:
                            # 필요한 열들만 추출 (2열 ~ 7열)
                            selected_data = [float(values[i]) for i in range(1, 7)]
                            file_data_list.append(selected_data)
                        except ValueError:
                            print(f"잘못된 데이터 형식 무시: {line.strip()}")
        
        # NumPy 배열로 변환 후 추가
        file_data_array = np.array(file_data_list)
        data_list.extend(file_data_list)
        
        # 파일별 데이터 크기 저장
        data_sizes[file_name] = file_data_array.shape
    
    # 최종 데이터 배열로 변환
    total_data_array = np.array(data_list)
    return total_data_array

# 데이터 로드 및 전처리 실행
data_array = load_and_preprocess_data()

# 각 파일별 데이터 크기 출력
print("\n===== 개별 파일 데이터 크기 확인 =====")
for file_name, size in data_sizes.items():
    print(f"{file_name}: {size}")

# 전체 데이터 크기 출력
print("\n===== 전체 데이터 크기 =====")
print(f"Total Data Size: {data_array.shape}")
