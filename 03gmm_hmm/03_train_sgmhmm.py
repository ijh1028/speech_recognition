# 03_train_gmmhmm.pt : GMM-HMM 학습 코드

from hmmfunc import MonoPhoneHMM

import numpy as np
import sys
import os

if __name__ == "__main__":

    # 학습원의 HMM파일
    base_hmm = './exp/model_3state_1mix/0.hmm'

    # 훈련 데이터의 특징량 리스트 파일
    feat_scp = \
        '../01compute_features/mfcc/train_small/feats.scp'

    # 훈련 데이터의 라벨 파일
    label_file = \
        './exp/data/train_small/text_int'

    # 학습 결과를 저장할 폴더
    work_dir = './exp'

    # 갱신 횟수
    num_iter = 10

    # 혼합 수를 증가시킨 횟수, 증가시킬 때마다 혼합 수는 2배가 된다
    # 최종적인 혼합 수는 2^(mixup_time)이 된다
    # 갱신 횟수는 num_iter * (mixup_time + 1)
    mixup_time = 1

    # 학습에 사용 언어 수
    # 실제로는 모든 언어를 사용하지만, 시간이 걸리기 때문에 이 프로그램에서는 일부의 언어만을 사용하고 있다
    num_utters = 50

    # 학습 전의 HMM을 읽기
    hmm = MonoPhoneHMM()
    hmm.load_hmm(base_hmm)

    # 학습 대상 HMM 파일 상태 수를 구한다
    num_states = hmm.num_states
    # 학습 대상 HMM 파일 혼합 수를 구한다
    num_mixture = hmm.num_mixture

    # 라벨 파일을 열어 발화 ID별 라벨 정보를 구한다
    label_list = {}
    with open(label_file, mode='r') as f:
        for line in f:
            # 0번째 줄은 언어 ID
            utt = line.split()[0]
            # 1열 이후에는 라벨
            lab = line.split()[1:]
            # 각 요소는 문자로서 읽혀져 있기 때문에 정수값으로 변환하다
            lab = np.int64(lab)
            # label_list에 등록
            label_list[utt] = lab

        # 특징량 리스트 파일을 열고 발화 ID별 특징량 파일의 경로를 얻다
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        # 특징량의 경로를 feat_list에 추가해 나가다
        # 이때 학습에 사용하는 발화 몇 분만 추가한다(모든 데이터를 학습에 사용하면 시간이 걸리기 때문)
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            # 0열은 발화 ID
            utt = line.split()[0]
            # 첫 번째 줄은 파일 경로
            ff = line.split()[1]
            # 세 번째 줄은 차원수
            nd = int(line.split()[3])
            # 발화 ID가 label_에 존재하지 않으면 에러
            if not utt in label_list:
                sys.stderr.write( \
                    '%s does not have label\n' % (utt))
                exit(1)
            # 차원수가 HMM의 차원수와 일치하지 않으면 에러
            if hmm.num_dims != nd:
                sys.stderr.write( \
                    '%s: unexpected #dims (%d)\n' \
                    % (utt, nd))
                exit(1)
            # feat_file 등록
            feat_list[utt] = ff

    # 출력 디렉토리 이름
    model_name = 'model_%dstate_%dmix' \
                 % (num_states, num_mixture)
    out_dir = os.path.join(work_dir, model_name)

    # 출력 디렉토리가 존재하지 않는 경우 작성
    os.makedirs(out_dir, exist_ok=True)

    #혼합 수 증가 횟수만큼 루프
    for m in range(mixup_time + 1):
        # 혼합 수 증가에 대한 과정을 수행
        # 신규 폴더에 0.hmm 이름으로 저장
        if m > 0:
            # 혼합 수 증가 과정을 수행
            hmm.mixup()
            # 혼합 수를 2배로 한다
            num_mixture *= 2

            # 출력 디렉토리 이름
            model_name = 'model_%dstate_%dmix' \
                         % (num_states, num_mixture)
            out_dir = os.path.join(work_dir, model_name)

            # 출력 디렉토리가 존재하지 않는 경우 작성
            os.makedirs(out_dir, exist_ok=True)

            # HMM 파일 저장
            out_hmm = os.path.join(out_dir, '0.hmm')
            hmm.save_hmm(out_hmm)
            print('Increased mixtures %d -> %d' \
                  % (num_mixture / 2, num_mixture))
            print('saved model: %s' % (out_hmm))

        # GMM 혼합 수를 기준으로 지정된  num_iter 횟수만큼 반복
        for iter in range(num_iter):
            print('%d-th iterateion' % (iter + 1))
            # 학습(1iteration분)
            hmm.train(feat_list, label_list)

            # HMM 프로토타입을 json 형식으로 저장
            out_hmm = os.path.join(out_dir,
                                   '%d.hmm' % (iter + 1))
            # 학습한 HMM 저장
            hmm.save_hmm(out_hmm)
            print('saved model: %s' % (out_hmm))