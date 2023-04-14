# 06_recognize.py : 고립단어 음성인식을 실행하는 메인 코드

from hmmfunc import MonoPhoneHMM

import numpy as np
import sys
import os

if __name__ == "__main__":
    # HMM 파일
    hmm_file = './exp/model_3state_1mix/10.hmm'

    # 평가 데이터 특징값 목록 파일
    feat_scp = './exp/data/test/mfcc/feats.scp'

    # 사전 파일
    lexicon_file = './exp/data/test/lexicon.txt'

    # 음소 목록
    phone_list_file = './exp/data/train_small/phone_list'

    # True일 경우, 문서 시작과 끝에 공백(pause)이 있음을 가정
    insert_sil = True

    # 음소 목록 파일을 열고, phone_list 변수에 저장
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 음소 목록 파일에서 음소를 구한다
            phone = line.split()[0]
            # 음소 목록 끝에 추가
            phone_list.append(phone)

    # 사전 파일을 열어, 단어와 음소 배열 대응 목록을 구한다
    lexicon = []
    with open(lexicon_file, mode='r') as f:
        for line in f:
            # 0번째 값은 단어
            word = line.split()[0]
            # 1번째 값 이후는 음소 배열
            phones = line.split()[1:]
            # insert_sil이 True일 경우에는 파일 양끝에 pause 추가
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
            # 단어, 음소 배열, 수치 값을 사전으로 변화하여 lexicon에 추가
            lexicon.append({'word': word,
                            'pron': phones,
                            'int': ph_int})

    # HMM을 읽기
    hmm = MonoPhoneHMM()
    hmm.load_hmm(hmm_file)

    # 특징값 목록 파일을 열어 발화별로 음성 인식을 수행
    with open(feat_scp, mode='r') as f:
        for line in f:
            # 0번째는 발화ID, 1번째는 파일경로
            utt = line.split()[0]
            ff = line.split()[1]
            # 3번째는 차원수
            nd = int(line.split()[3])

            # 차원수가 HMM의 차원수와 일치하지 않으면 에러
            if hmm.num_dims != nd:
                sys.stderr.write( \
                    '%s: unexpected #dims (%d)\n' \
                    % (utt, nd))
                exit(1)

            # 특징값 파일 열기
            feat = np.fromfile(ff, dtype=np.float32)
            # 프레임 수 x 차원 수 배열로 변형
            feat = feat.reshape(-1, hmm.num_dims)

            # 고립단어 인식을 수행
            (result, detail) = hmm.recognize(feat, lexicon)

            # result에는 가장 빈도가 높은 단어가 저장되어 있고
            # detail에는 빈도의 순위값이 저장되어 있는 결과를 출력
            sys.stdout.write('%s %s\n' % (utt, ff))
            sys.stdout.write('Result = %s\n' % (result))
            sys.stdout.write('[Runking]\n')
            for res in detail:
                sys.stdout.write('  %s %f\n' \
                                 % (res['word'], res['score']))