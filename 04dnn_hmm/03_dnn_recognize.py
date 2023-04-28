# 03_dnn_recognize.py : DNN-HMM 기반 고립단어 음성인식을 수행하는 코드

import torch
import torch.nn.functional as F

# 작성한 Dataset 클래스 가져오기
from my_dataset import SequenceDataset

# hmmfunc.py 에서 MonoPhoneHMM 클래스 가져오기
from hmmfunc import MonoPhoneHMM

# 직접 정의한 DNN 모델 불러오기
from my_model import MyDNN

import numpy as np
import json
import os
import sys

if __name__ == "__main__":

    # 평가 데이터 특징값 목록
    test_feat_scp = '../03gmm_hmm/exp/data/test/mfcc/feats.scp'

    # HMM 파일
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # DNN 모델 파일
    dnn_file = './exp/model_dnn/best_model.pt'

    # HMM 상태 출현 카운트 파일
    count_file = './exp/model_dnn/state_counts'

    # 학습 데이터에서 계산된 특징값 평균 / 표준편차 파일
    mean_std_file = 'exp/model_dnn/mean_std.txt'

    # 사전 파일
    lexicon_file = '../03gmm_hmm/exp/data/test/lexicon.txt'

    # 음소 리스트
    phone_list_file = '../03gmm_hmm/exp/data/train_small/phone_list'

    # True일 경우 문장 선두와 마지막에 pause가 있는 것을 전제로 한다
    insert_sil = True

    # DNN 학습 시 출력한 설정 파일
    config_file = os.path.join( \
        os.path.dirname(dnn_file),
        'config.json')

    # HMM 읽어온다
    hmm = MonoPhoneHMM()
    hmm.load_hmm(hmm_file)

    # 설정 파일 가져오기
    with open(config_file, mode='r') as f:
        config = json.load(f)

    # 가져온 설정을 반영하다
    # 은닉층 Layer 개수
    num_layers = config['num_layers']
    # 은닉층 차원수
    hidden_dim = config['hidden_dim']
    # splice 프레임 수
    splice = config['splice']

    # 특징값 평균 / 표준편차 파일을 읽어온다
    with open(mean_std_file, mode='r') as f:
        # 全行読み込み
        lines = f.readlines()
        # 1행(0시작)가 평균값 벡터(mean), 3행은 표준편차벡터(std)
        mean_line = lines[1]
        std_line = lines[3]
        # 공백으로 구분하여 리스트로 변환
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy array로 변환
        feat_mean = np.array(feat_mean,
                             dtype=np.float32)
        feat_std = np.array(feat_std,
                            dtype=np.float32)

    # 차원 수 정보를 구하기
    feat_dim = np.size(feat_mean)

    # DNN 출력층의 차원 수는 음소 수 * 상태 수
    dim_out = hmm.num_phones * hmm.num_states

    # Neural Network 모델을 생성
    # 입력 특징값의 차원 수는 feat_dim * (2*splice+1)
    dim_in = feat_dim * (2 * splice + 1)
    model = MyDNN(dim_in=dim_in,
                  dim_hidden=hidden_dim,
                  dim_out=dim_out,
                  num_layers=num_layers)

    # 학습을 마친 DNN 파일에서 모델 매개변수를 읽어온다
    model.load_state_dict(torch.load(dnn_file))

    # 모델을 평가 모드로 설정
    model.eval()

    # HMM 상태 카운트 파일을 읽어온다
    with open(count_file, mode='r') as f:
        # 1행을 읽어온다
        line = f.readline()
        # HMM 상태별 출현 횟수가 공백으로 구분되어
        # 입력되므로 공백으로 구분해서 리스트를 만든다
        count = line.split()
        # 각 수치는 문자형이므로 수치로 변환
        count = np.float32(count)

        # 전체합으로 나누어, 각 HMM 상태의 사전 발생 확률로 변환
        prior = count / np.sum(count)

    # 음소 목록 파일을 열어 phone_list에 저장
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 음소 목록 파일에서 음소를 구한다
            phone = line.split()[0]
            # 음소 목록 맨 마지막에 추가
            phone_list.append(phone)

    # 사전 파일을 열어 단어와 음소열의 대응 목록을 구한다
    lexicon = []
    with open(lexicon_file, mode='r') as f:
        for line in f:
            # 0열은 단어
            word = line.split()[0]
            # 1열 이후에는 음소열
            phones = line.split()[1:]
            # insert_sil이 True인 경우 양쪽 끝에 포즈 추가
            if insert_sil:
                phones.insert(0, phone_list[0])
                phones.append(phone_list[0])
            # phone_list를 사용하여 음소를 수치로 변환
            ph_int = []
            for ph in phones:
                if ph in phone_list:
                    ph_int.append(phone_list.index(ph))
                else:
                    sys.stderr.write('invalid phone %s' % (ph))
            # 단어, 음소열, 수치표기 사전으로서 lexicon에 추가
            lexicon.append({'word': word,
                            'pron': phones,
                            'int': ph_int})

    # 특징값 목록 파일을 연다
    with open(test_feat_scp, mode='r') as f:
        for line in f:
            # 발화 ID, 파일 경로, 차원 수를 구한다
            utt = line.split()[0]
            ff = line.split()[1]
            nd = int(line.split()[3])

            # 특징값 데이터를 읽어온다
            feat = np.fromfile(ff, dtype=np.float32)
            # 프레임 수 x 차원 수 배열로 변형
            feat = feat.reshape(-1, nd)

            # 평균과 표준편차를 사용하여 정규화(표준화)를 수행
            feat = (feat - feat_mean) / feat_std

            # splicing: 전후 n프레임 특징값을 결합
            org_feat = feat.copy()
            for n in range(-splice, splice + 1):
                # 원래 특징을 n프레임 이동시킨다
                tmp = np.roll(org_feat, n, axis=0)
                if n < 0:
                    # 앞으로 이동한 경우 맨 마지막 n개 프레임을 0으로 한다
                    tmp[n:] = 0
                elif n > 0:
                    # 뒤로 이동한 경우 처음 n개 프레임을 0으로 한다
                    tmp[:n] = 0
                else:
                    continue
                # 이동시킨 특징값을 차원축으로 결합
                feat = np.hstack([feat, tmp])

            # pytorch의 DNN 모델에 입력하기 위해 torch.tensor 형태로 변환
            feat = torch.tensor(feat)

            # DNN에 입력
            output = model(feat)

            # softmax 함수로 확률값을 구한다
            output = F.softmax(output, dim=1)

            # numpy array형으로 변환
            output = output.detach().numpy()

            # 각 HMM 상태의 사전 발생률로 나누고 로그를 취하여 로그 빈도로 변환하다
            # (HMM 각 상태별 출력 확률에 해당하는 형태로 변환)
            likelihood = np.log(output / prior)
            
            # 로그 빈도를 사용하여 HMM 기반 음성인식을 수행
            (result, detail) = hmm.recognize_with_dnn(likelihood, lexicon)
            # 결과를 출력
            sys.stdout.write('%s %s\n' % (utt, ff))
            sys.stdout.write('Result = %s\n' % (result))
            sys.stdout.write('[Runking]\n')
            for res in detail:
                sys.stdout.write('  %s %f\n' \
                                 % (res['word'], res['score']))