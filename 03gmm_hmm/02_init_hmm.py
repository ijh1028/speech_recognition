# 02_init_HMM.py : HMM 매개변수를 초기화하는 메인 코드

# HMMfunc.py에서 MonoPhoneHMM클래스를 가져옵니다
from hmmfunc import MonoPhoneHMM

import numpy as np

import os

if __name__ == "__main__":
    # HMM 프로토타입
    hmmproto = './exp/model_3state_1mix/hmmproto'
    # 학습 데이터 특징값 평균/표준편차 파일
    mean_std_file = '../01compute_features/mfcc/train_small/mean_std.txt'
    # 출력 파일 위치
    out_dir = os.path.dirname(hmmproto)

    # 출력 디렉터리가 없는 경우 작성
    os.makedirs(out_dir, exist_ok=True)

    # 특징값 평균/표준편차 파일 읽어오기
    with open(mean_std_file, mode='r') as f:
        # 모든 행을 읽어오기
        lines = f.readlines()
        # 1행째(0시작)가 평균값 벡터(mean)
        # 3행째가 표준편차 벡터(std)
        mean_line = lines[1]
        std_line = lines[3]
        # 공백으로 구분된 리스트로 변환
        mean = mean_line.split()
        std = std_line.split()
        # numpy array로 변환
        mean = np.array(mean, dtype=np.float64)
        std = np.array(std, dtype=np.float64)
        # 표준편차를 분산으로 변환
        var = std ** 2

    # MonoPhoneHMM 클래스를 호출한다
    hmm = MonoPhoneHMM()
    # HMM 프로토타입을 읽어온다
    hmm.load_hmm(hmmproto)
    # Flat Start 초기화 실행
    hmm.flat_init(mean, var)
    # HMM 프로토타입을 json 형식으로 저장
    hmm.save_hmm(os.path.join(out_dir, '0.hmm'))
