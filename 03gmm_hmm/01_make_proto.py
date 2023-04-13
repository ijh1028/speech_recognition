# 01_make_proto.py : HMM 프로토타입을 생성하는 Main Program

import os

# hmmfunc.py 에서 MonoPhoneHMM 클래스를 import
from hmmfunc import MonoPhoneHMM

if __name__ == "__main__":
    # 음소 목록
    phone_list_file = './exp/data/train_small/phone_list'

    # 각 음소의 HMM 상태 수
    num_states = 3
    # 입력 특징값 차원 수
    # 여기서는 MFCC를 사용하므로 MFCC 차원 수를 입력
    num_dims = 13
    # 자기 루프 확률 초기값
    prob_loop = 0.7
    # 출력 폴더
    out_dir = './exp/model_%dstate_1mix' % (num_states)

    # 음소 목록 파일을 열어 .phone_list에 저장
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 음소 목록 파일에서 음소 기록
            phone = line.split()[0]
            # 음소 목록 끝에 추가
            phone_list.append(phone)

    # MonoPhoneHMM 클래스를 생성
    hmm = MonoPhoneHMM()
    # HMM 프로토타입을 생성
    hmm.make_proto(phone_list, num_states,
                   prob_loop, num_dims)
    # HMM 프로토타입을 json 형식으로 저장
    hmm.save_hmm(os.path.join(out_dir, 'hmmproto'))