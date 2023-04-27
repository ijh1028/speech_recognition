# my_model.py : DNN-HMM에 사용되는 DNN모델

import torch.nn as nn

# 작성한 initialize.py에서 초기화 함수
# lecun_initialization을 불러온다
from initialize import lecun_initialization

class MyDNN(nn.Module):
    ''' Fully connected layer (선형층) 기반 단순 DNN
    dim_in:     입력 특징값 차원 수
    dim_hidden: 은닉층 차원 수
    dim_out:    출력 차원 수
    num_layers: 은닉층 개수
    '''

    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 num_layers=2):
        super(MyDNN, self).__init__()

        # 은닉층 수
        self.num_layers = num_layers

        # 입력층 : 선형층 + ReLU
        self.inp = nn.Sequential( \
            nn.Linear(in_features=dim_in,
                      out_features=dim_hidden),
            nn.ReLU())
        # 은닉층
        hidden = []
        for n in range(self.num_layers):
            # 선형층을 hidden에 첨가
            hidden.append(nn.Linear(in_features=dim_hidden,
                                    out_features=dim_hidden))
            # ReLU를 추가
            hidden.append(nn.ReLU())

        # Pytorch에서 다루기 위해서 리스트를 ModuleList로 변환
        self.hidden = nn.ModuleList(hidden)

        # 출력층 : 선형층
        self.out = nn.Linear(in_features=dim_hidden,
                             out_features=dim_out)

        # LeCun 매개변수 초기화 실행
        lecun_initialization(self)

    def forward(self, frame):
        ''' 순전달파(forward처리)함수
        frame:  입력 프레임 데이터
        output: 입력된 프레임에 대한 상태 확률
        '''
        # 입력층을 통과
        output = self.inp(frame)
        # 은닉층을 통과
        for n in range(self.num_layers):
            output = self.hidden[n](output)
        # 출력층을 통과
        output = self.out(output)
        return output