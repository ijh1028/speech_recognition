#  00_state_alignment.py : 음소 얼라인먼트를 추정하는 코드

from hmmfunc import MonoPhoneHMM

import numpy as np
import sys
import os

if __name__ == "__main__":

    # HMM 파일
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # 특징값 목록 파일
    train_feat_scp = \
        '../01compute_features/mfcc/train_small/feats.scp'
    # 개발 데이터 특징값 목록 파일
    dev_feat_scp = \
        '../01compute_features/mfcc/dev/feats.scp'

    # 훈련 데이터의 라벨 파일
    train_label_file = \
        '../03gmm_hmm/exp/data/train_small/text_int'
    # 개발 데이터 라벨 파일
    dev_label_file = \
        '../03gmm_hmm/exp/data/dev/text_int'

    # 훈련 데이터 정렬 결과 출력 파일
    train_align_file = \
        './exp/data/train_small/alignment'
    # 개발 데이터 정렬 결과 출력 파일
    dev_align_file = \
        './exp/data/dev/alignment'

    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # 학습 전의 HMM을 읽기
    hmm.load_hmm(hmm_file)

    # 훈련/개발 데이터의 특징값/라벨/배열 파일을 목록화
    feat_scp_list = [train_feat_scp, dev_feat_scp]
    label_file_list = [train_label_file, dev_label_file]
    align_file_list = [train_align_file, dev_align_file]

    for feat_scp, label_file, align_file in \
            zip(feat_scp_list, label_file_list, align_file_list):

        # 출력 디렉토리
        out_dir = os.path.dirname(align_file)

        # 출력 디렉토리가 존재하지 않는 경우 작성
        os.makedirs(out_dir, exist_ok=True)

        # 라벨 파일을 열고 발화ID별 라벨 정보를 얻다
        label_list = {}
        with open(label_file, mode='r') as f:
            for line in f:
                # 0열은 발화ID
                utt = line.split()[0]
                # 1열 이후에는 라벨
                lab = line.split()[1:]
                # 각 요소는 문자로서 읽혀져 있기 때문에 정수값으로 변환하다
                lab = np.int64(lab)
                # label_list에 등록
                label_list[utt] = lab

        # 발화마다 얼라인먼트를 추정
        with open(align_file, mode='w') as fa, \
                open(feat_scp, mode='r') as fs:
            for line in fs:
                # 0열은 발화ID
                utt = line.split()[0]
                print(utt)
                # 1열 이후에는 라벨
                ff = line.split()[1]
                # 3열은 차원수
                nd = int(line.split()[3])

                # 발화ID가 label_에 존재하지 않으면 에러
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

                # 라벨을 얻기
                label = label_list[utt]
                # 특징값 파일 열기
                feat = np.fromfile(ff, dtype=np.float32)
                # 프레임 수 x 차원 수 배열로 변형
                feat = feat.reshape(-1, hmm.num_dims)

                # 정렬 실행
                alignment = hmm.state_alignment(feat, label)
                # alignment는 수치 목록이므로 파일에 쓰기 위해 문자열로 변환하다
                alignment = ' '.join(map(str, alignment))
                # 파일로 출력
                fa.write('%s %s\n' % (utt, alignment))