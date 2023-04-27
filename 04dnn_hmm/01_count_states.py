# 01_count_states.py : 각 HMM 상태의 출현 빈도를 세어본다

from hmmfunc import MonoPhoneHMM

import numpy as np
import sys
import os

if __name__ == "__main__":
    # HMM 파일 경로
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # 훈련 데이터의 얼라인먼트 파일
    align_file = './exp/data/train_small/alignment'

    # 미리 계산한 사전 확률 파일 경로
    count_file = './exp/model_dnn/state_counts'

    # 학습 전에 HMM을 읽기
    hmm = MonoPhoneHMM()
    hmm.load_hmm(hmm_file)

    # HMM 모든 상태 수를 구하기
    num_states = hmm.num_phones * hmm.num_states

    # 출력 디렉토리
    out_dir = os.path.dirname(count_file)

    # 출력 디렉토리가 존재하지 않는 경우 작성
    os.makedirs(out_dir, exist_ok=True)

    # 상태별 출력 카운터
    count = np.zeros(num_states, np.int64)

    # 얼라인먼트 파일 열기
    with open(align_file, mode='r') as f:
        for line in f:
            # 0번째는 발화ID
            utt = line.split()[0]
            # 1번째 이후는 얼라인먼트
            ali = line.split()[1:]
            # 얼라인먼트는 문자 형태로 읽어오므로 정수값으로 변환
            ali = np.int64(ali)
            # 상태값을 하나씩 읽어오며 카운터를 1씩 증가
            for a in ali:
                count[a] += 1

    # 카운트가 0인 것은 1로 한다
    # 이후 처리에서 0으로 나누는 것을 방지하기 위함
    count[count==0] = 1

    # 카운트 결과를 저장
    with open(count_file, mode='w') as f:
        # 벡터 count를 문자열로 변환
        count_str = ' '.join(map(str, count))
        f.write('%s\n' % (count_str))