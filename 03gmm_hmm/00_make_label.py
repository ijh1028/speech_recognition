# make_label.py : 라벨 파일 음소 표기를 문자에서 숫자로 변환

import os
import sys

def phone_to_int(label_str, 
                 label_int,
                 phone_list, 
                 insert_sil=False):
    '''
    음소 목록을 사용하여 라벨 파일 음소를 숫자로 변환
    label_str : 문자로 기술된 라벨 파일
    label_int : 문자를 숫자로 변환한 후 라벨 파일 저장 경로
    phone_list : 음소목록
    insrt_sil : True인 경우 텍스트 시작과 끝에 공백을 삽입
    '''

    # 파일 열기
    with open(label_str, mode='r') as f_in, open(label_int, mode='w') as f_out:
        # 라벨 파일을 1행씩 읽기
        for line in f_in:
            # 읽어온 행을 공백으로 구분하여 목록형 변수로 만들기
            text = line.split()

            # 목록 0번째 요소는 발화 ID라서 그대로 출력
            f_out.write('%s' % text[0])

            # insert_sil이 True일 경우 시작 지점에 0(pause) 삽입
            if insert_sil:
                f_out.write(' 0')
            # 첫 번째 목록 이후의 요소는 문자이기 때문에 1문자씩 숫자로 치환
            for u in text[1:]:
                if not u in phone_list:
                    sys.stderr.write('phone_to_int: \
                        unknown phone %s\n' % u)
                    exit(1)
                # 음소 Index 출력
                f_out.write(' %d' % (phone_list.index(u)))

            # insert_sil이 true일 경우 마지막 지점에 0(Pause) 삽입
            if insert_sil:
                f_out.write(' 0')
            # 개행
            f_out.write('\n')

if __name__ == "__main__":
    # 훈련 데이터의 라벨 파일 경로
    label_train_str = '../data/label/train_small/text_phone'

    # 훈련 데이터의 초기 결과를 저장할 디렉토리 경로
    out_train_dir = './exp/data/train_small'

    # 개발 데이터 라벨 파일 경로(개발 데이터는 GMM-HMM에는 사용하지 않습니다만, DNN-HMM에서 사용합니다．)
    label_dev_str = \
        '../data/label/dev/text_phone'

    # 개발 데이터 처리 결과 출력처 디렉토리
    out_dev_dir = \
        './exp/data/dev'   

    # 음소 목록
    phone_file = './phones.txt'

    # pause를 표시하는 기호
    silence_phone = 'pau'

    # true일 경우, 문장 시작과 끝에 pause를 삽입
    insert_sil = True

    # 음소 목록 앞에서  Pause 기호를 넣기
    phone_list = [silence_phone]
    # 음소 목록 파일을 열어 phone_list에 저장
    with open(phone_file, mode='r') as f:
        for line in f:
            # 공백과 개행을 제거하여 음소 기호 얻기
            phone = line.strip()
            # 음소 목록 끝에 추가
            phone_list.append(phone)

    # 훈련/개발 데이터 정보를 목록화
    label_str_list = [label_train_str,
                      label_dev_str]
    out_dir_list = [out_train_dir,
                    out_dev_dir]

    # 훈련/개발 데이터에 대해서 각각 처리
    for (label_str, out_dir) \
           in zip(label_str_list, out_dir_list):

        # 출력 디렉토리가 존재하지 않는 경우 작성
        os.makedirs(out_dir, exist_ok=True)


    # 음소와 숫자 대응 목록 출력
    out_phone_list = os.path.join(out_dir, 'phone_list')
    with open(out_phone_list, 'w') as f:
        for i, phone in enumerate(phone_list):
            # 목록에 등록되어 있는 순서를 해당 음소에 대응하는 숫자로 인식
            f.write('%s %d\n' % (phone, i))

    # 라벨의 음소 기호를 숫자로 변환하여 출력
    label_int = os.path.join(out_dir, 'text_int')
    phone_to_int(label_str, 
                 label_int,
                 phone_list, 
                 insert_sil)