import numpy as np
import os
import torch
from torch.utils.data import TensorDataset

from collections import OrderedDict


# 데이터 전처리 함수
def preprocess_data(data):
    # 데이터를 텐서로 변환
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor


# 데이터셋 생성 함수
def create_dataset(root_dir, mode=None):              
    data = []
    labels = []
    for class_dir in sorted(os.listdir(root_dir), key=lambda x: int(x[1:])):         
        class_path = os.path.join(root_dir, class_dir)    
        for csv_file in sorted(os.listdir(class_path), key=lambda x: int(x[:-4])):       
            csv_path = os.path.join(class_path, csv_file) 
            # CSV 파일 읽기
            vec = np.loadtxt(csv_path)                  
            # 데이터 전처리
            data_tensor = preprocess_data(vec)          
            data.append(data_tensor)                   
            # 레이블 추가 (클래스 이름)
            labels.append(class_dir)                    

    # 데이터를 하나의 텐서로 병합
    data_tensor = torch.stack(data)                       
    # 레이블을 클래스 인덱스로 변환
    class_to_index = {class_name: i for i, class_name in enumerate(list(OrderedDict.fromkeys(labels)))}
    labels = [class_to_index[label] for label in labels]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    print(f"{mode} datasets complete!")

    return TensorDataset(data_tensor, labels_tensor)