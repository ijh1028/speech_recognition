# my_dataset.py : 음성 특징값과 라벨 데이터를 취급하는 SequenceDataset 클래스

# 파이토치의 Dataset 모듈을 읽어오기
from torch.utils.data import Dataset

import numpy as np
import sys

class SequenceDataset(Dataset):
    ''' MiniBatch 데이터를 생성하는 클래스
        torch.utils.data.Dataset 클래스를 계승하여 아래 함수를 정의
        __len__: 총 샘플 수를 출력하는 함수
        __getitem__: 단일 샘플 데이터를 출력하는 함수
    feat_scp:  특징값 목록 파일
    label_scp: 라벨 파일
    feat_mean: 특징값의 평균값 벡터
    feat_std:  특징 차원별 표준편차를 나열한 벡터
    pad_index: 데이터를 배치로 처리할 때 프레임 수를 맞추기 위한 padding에 대한 정수값
    splice:    전후(splice) 프레임을 특징값으로 결합하는 값
               splice = 1의 경우, 전후 1개 프레임을 결합하므로 차원 수는 3배가 된다
               splice = 0의 경우, 아무 처리도 하지 않는다
    '''
    def __init__(self,
                 feat_scp,
                 label_scp,
                 feat_mean,
                 feat_std,
                 pad_index,
                 splice=0):
        # 발화 개수
        self.num_utts = 0
        # 각 발화 ID
        self.id_list = []
        # 각 발화 특징값 파일의 경로를 기입할 리스트
        self.feat_list = []
        # 각 발화의 특징값 프레임 개수를 기입할 리스트
        self.feat_len_list = []
        # 특징값 평균값 벡터
        self.feat_mean = feat_mean
        # 특징값 표준편차 벡터
        self.feat_std = feat_std
        # 표준편차 플로어링
        # (0으로 나누는 일이 발생하지 않기 위함)
        self.feat_std[self.feat_std<1E-10] = 1E-10
        # 특징값 차원 수
        self.feat_dim = \
            np.size(self.feat_mean)
        # 각 발화 라벨
        self.label_list = []
        # 각 발화 라벨 길이를 기입할 리스트
        self.label_len_list = []
        # 프레임 개수 최댓값
        self.max_feat_len = 0
        # 라벨 길이 최댓값
        self.max_label_len = 0
        # 프레임을 채우기 위해 사용되는 정수값
        self.pad_index = pad_index
        # splice: 전후 n개 프레임 특징값을 결합
        self.splice = splice

        # 특징값 목록, 라벨을 1행씩 읽어 나가면서 정보를 저장
        with open(feat_scp, mode='r') as file_f, \
             open(label_scp, mode='r') as file_l:
            for (line_feats, line_label) in zip(file_f, file_l):
                # 각 행을 공백으로 구분하여 목록형 변수로 변환
                parts_feats = line_feats.split()
                parts_label = line_label.split()

                # 발화ID(parts의 0번째 요소)가 특징량과 라벨이 일치하지 않으면 에러
                if parts_feats[0] != parts_label[0]:
                    sys.stderr.write('IDs of feat and '\
                        'label do not match.\n')
                    exit(1)

                # 발화 ID를 리스트에 추가
                self.id_list.append(parts_feats[0])
                # 특징 파일 경로를 리스트에 추가
                self.feat_list.append(parts_feats[1])
                # 프레임 수를 리스트에 추가
                feat_len = np.int64(parts_feats[2])
                self.feat_len_list.append(feat_len)

                # 라벨(번호로 기입)을 int형의 numpy array로 변환
                label = np.int64(parts_label[1:])
                # 라벨 리스트에 추가
                self.label_list.append(label)
                # 라벨 길이를 추가
                self.label_len_list.append(len(label))

                # 발화 수를 카운팅
                self.num_utts += 1

        # 프레임 개수를 최댓값을 구한다
        self.max_feat_len = \
            np.max(self.feat_len_list)
        # 라벨 길이를 최댓값을 구한다
        self.max_label_len = \
            np.max(self.label_len_list)

        # 라벨 데이터 길이를 최대 프레임 길이에 맞추기 위해 pad_index
        for n in range(self.num_utts):
            # 채우는 프레임 개수 = 최대 프레임 수 - 자신의 프레임 수
            pad_len = self.max_label_len \
                    - self.label_len_list[n]
            # pad_index 값으로 채운다
            self.label_list[n] = \
                np.pad(self.label_list[n],
                       [0, pad_len],
                       mode='constant',
                       constant_values=self.pad_index)

    def __len__(self):
        ''' 학습 데이터의 총 샘플 수를 반환하는 함수
        이번 구현에서는 발화 단위로 배치를 생성하므로
        총 샘플 수 = 발화 개수
        '''
        return self.num_utts


    def __getitem__(self, idx):
        ''' 샘플 데이터를 반환하는 함수
        이번 구현에서는 발화 단위로 배치를 생성하므로
        idx = 발화 번호
        '''
        # 특징값 계열의 프레임 개수
        feat_len = self.feat_len_list[idx]
        # 라벨 길이
        label_len = self.label_len_list[idx]

        # 특징값 데이터를 특징값 파일로부터 읽어오기
        feat = np.fromfile(self.feat_list[idx],
                           dtype=np.float32)
        # 프레임 수 * 차원 수 배열로 변형
        feat = feat.reshape(-1, self.feat_dim)

        # 평균과 표준편차를 사용하여 정규화(표준화)를 수행
        feat = (feat - self.feat_mean) / self.feat_std

        # splicing: 전후 n개 프레임의 특징값을 결합
        org_feat = feat.copy()
        for n in range(-self.splice, self.splice+1):
            # 기존 특징값을 n 프레임 이동
            tmp = np.roll(org_feat, n, axis=0)
            if n < 0:
                # 앞으로 이동했을 경우, 마지막 n개 프레임을 0으로 한다
                tmp[n:] = 0
            elif n > 0:
                # 뒤로 이동했을 경우, 첫 n개 프레임을 0으로 한다
                tmp[:n] = 0
            else:
                continue
            # 이동한 특징값을 차원 방향으로 결합
            feat = np.hstack([feat,tmp])

        # 특징값 데이터의 프레임 개수를 최대 프레임 개수에 더하기 위해 0으로 채운다
        pad_len = self.max_feat_len - feat_len
        feat = np.pad(feat,
                      [(0, pad_len), (0, 0)],
                      mode='constant',
                      constant_values=0)

        # 라벨
        label = self.label_list[idx]

        # 발화 ID
        utt_id = self.id_list[idx]

        # 특징값, 라벨, 프레임 개수, 라벨 길이, 발화 ID를 반환
        return (feat,
               label,
               feat_len,
               label_len,
               utt_id)