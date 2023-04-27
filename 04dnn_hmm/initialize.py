# initialize.py : LeCun 기반 매개변수 초기화 실행

import numpy as np

def lecun_initialization(model):
    '''LeCun 기반 매개변수 초기화 방법 실행 함수
    각 매개변수(bias 제외)를 평균 0, 표준편차 1/ sqrt(dim)의
    정규분포를 따르는 난수로 초기화(dim은 입력 차원 수)
    model: Pytorch에서 정의한 모델
    '''

    # 모델 매개변수를 순서대로 추출하여 초기화
    for param in model.parameters():
        # 매개변수 값을 추출
        data = param.data
        # 매개변수 텐서의 차원 수를 추출
        dim = data.dim()
        # 차원 수에 따라 처리를 다르게 한다
        if dim == 1:
            # dim = 1바이어스 성부 처리
            # 0으로 초기화
            data.zero_()
        elif dim == 2:
            # dim = 2일 경우, 선형 함수의 매개변수
            # 입력 차원 수 = size(1)을 취득
            n = data.size(1)
            # 입력 차원 수의 제곱근 역수를 표준편차로 하는 정규분포를 따르는 난수로 초기화
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        elif dim == 3:
            # dim = 3의 경우 1차원 접이식 행렬
            # 입력 채널 수 * 커널 크기의 제곱근의 역수를 표준 편차로 하다
            # 정규 분포 난수로 초기화
            n = data.size(1) * data.size(2)
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        elif dim == 4:
            # dim = 4의 경우는 2차원 접이식 행렬
            # 입력 채널 수 * 커널 크기(행) * 커널사이즈(열)의 제곱근의 역수를 표준 편차로 하다
            # 정규 분포 난수로 초기화
            n = data.size(1) * data.size(2) * data.size(3)
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        else:
            # 그 이외에는 대응하고 있지 않다
            print('lecun_initialization: '\
                  'dim > 4 is not supported.')
            exit(1)