# -*- coding: utf-8 -*-

# 데이터 리스트를, 학습/개발/평가 데이터 세트로 분할
# BASIC5000_0001~0250 : 평가 데이터
# BASIC5000_0251~0500 : 개발 데이터
# BASIC5000_0501~1500 : 학습 데이터(소)
# BASIC5000_0501~5000 : 학습 데이터(대）

# os
import os

# 메인 함수
if __name__ == "__main__":
    
    # 전체 데이터의 저장 위치
    all_dir = 'D:/github/data/label/all'

    # 평가 데이터 출력
    out_eval_dir = 'D:/github/data/label/test'
    # 개발 데이터 출력
    out_dev_dir = 'D:/github/data/label/dev'
    # 학습 데이터(소) 출력
    out_train_small_dir = 'D:/github/data/label/train_small'
    # 학습 데이터(대) 출력
    out_train_large_dir = 'D:/github/data/label/train_large'

    # 각 출력 디렉토리가 존재하지 않는 경우 작성
    for out_dir in [out_eval_dir, out_dev_dir, 
                    out_train_small_dir, out_train_large_dir]:
        os.makedirs(out_dir, exist_ok=True)
    
    # wav.scp, text_char, text_kana, text_phone 각각 동일하게 처리
    for filename in ['wav.scp', 'text_char', 
                     'text_kana', 'text_phone']:
        # 평가/개발/학습 데이터 리스트를 각각 저장
        # 인코딩 오류 해결을 위해 'UTF-8' 사용
        with open(os.path.join(all_dir, filename), 
                  mode='r', encoding='UTF-8') as all_file, \
                  open(os.path.join(out_eval_dir, filename), 
                  mode='w', encoding='UTF-8') as eval_file, \
                  open(os.path.join(out_dev_dir, filename), 
                  mode='w', encoding='UTF-8') as dev_file, \
                  open(os.path.join(out_train_small_dir, filename), 
                  mode='w', encoding='UTF-8') as train_small_file, \
                  open(os.path.join(out_train_large_dir, filename), 
                  mode='w', encoding='UTF-8') as train_large_file:
            # 평가/개발/학습 데이터 리스트에 추가
            for i, line in enumerate(all_file):
                if i < 250:
                    # 1~250: 평가 데이터
                    eval_file.write(line)
                elif i < 500:
                    # 251~500: 개발 데이터
                    dev_file.write(line)
                else:
                    # 501~5000: 학습 데이터(소) 
                    train_large_file.write(line)
                    if i < 1500:
                        # 501～1500: 학습 데이터(대)
                        train_small_file.write(line)