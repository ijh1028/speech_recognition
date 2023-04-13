# 03_train_sgmhmm.py : SGM-HMM을 학습하는 메인 코드

from hmmfunc import MonoPhoneHMM

import numpy as np
import sys
import os

if __name__ == "__main__":
    # 학습할 HMM 파일
    base_hmm = './exp/model_3state_1mix/0.hmm'
    # 학습 데이터 특징값 목록 파일
    feat_scp = '../01compute_features/mfcc/train_small/feats.scp'
    # 학습 데이터 라벨 파일
    label_file = './exp/data/train_small/text_int'

    # 갱신 횟수
    num_iter = 10

    # 학습에 이용되는 발화 수, 실제로는 모든 발화를 이용하지만 시간 편의상 해당 코드에서는 일부 발화만 사용
    # 데이터는 총 1000개 발화 음성이 존재한다. ex) num)utters = 1000 도 가능하지만 시간이 오래 걸림
    num_utters = 50

    # 출력 디렉토리
    out_dir = os.path.dirname(base_hmm)

    # MonoPhoneHMM 클래스 호출
    hmm = MonoPhoneHMM()
    # 학습 전에 HMM 읽어오기
    hmm.load_hmm(base_hmm)

    # 라벨 파일을 열어 발화 ID별 라벨 정보를 구한다
    label_list = {}
    with open(label_file, mode='r') as f:
        for line in f:
            # 0번째 열은 발화 ID
            utt = line.split()[0]
            # 1번째 열 이후는 라벨
            lab = line.split()[1:]
            # 각 요소는 문자로 읽어들이므로 정수값으로 변환
            lab = np.int64(lab)
            # label_list에 등록
            label_list[utt] = lab

    # 특징값 목록 파일 열기, 발화 ID별로 특징값 파일 경로를 구한다
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        # 특징값 경로를 feat_list에 추가할 때, 학습에 사용하는 발화 수만큼 추가
        # (모든 데이터를 학습에 이용하기에는 시간이 너무 많이 소요)
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            # 0번째 열은 발화 ID, 1번째 열은 특징값 경로
            utt = line.split()[0]
            ff = line.split()[1]
            # feat file에 등록
            feat_list[utt] = ff

    # 학습 과정
    # num_iter 수만큼 반복하여 갱신
    for iter in range(num_iter):
        # 학습(literation의 내용)
        hmm.train(feat_list, label_list)
        # HMM 프로토타입 json 형식으로 저장
        out_hmm = os.path.join(out_dir,
                               '%d.hmm' % (iter+1))

        # 학습을 마친 HMM 저장
        hmm.save_hmm(out_hmm)
