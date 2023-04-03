# 다운로드한 명찰 데이터를 읽기，
# 문자(한자 섞인) 단위, 히라가나 단위, 음소 단위로 정의되는 라벨 파일을 작성

# yaml 데이터 가져오기
import yaml

# os
import os

# 메인 함수
if __name__ == "__main__":

    # 다운로드한 라벨데이터(yaml형식)
    original_label = '../data/original/jsut-label-master/text_kana/basic5000.yaml'

    # 라벨 리스트를 격납하는 장소
    out_label_dir = './data/label/all'

    # 출력 디렉터리가 없는 경우 작성
    os.makedirs(out_label_dir, exist_ok=True)

    # 라벨 데이터 열기
    # 인코딩 오류 해결을 위해 'UTF-8' 사용
    with open(original_label, mode='r', encoding='UTF-8') as yamlfile:
        label_info = yaml.safe_load(yamlfile)

    # 문자/히라가나/음소의 라벨 파일을 각각 저장
    # 인코딩 오류 해결을 위해 'UTF-8' 사용
    with open(os.path.join(out_label_dir, 'text_char'),
              mode='w', encoding='UTF-8') as label_char, \
            open(os.path.join(out_label_dir, 'text_kana'),
                 mode='w', encoding='UTF-8') as label_kana, \
            open(os.path.join(out_label_dir, 'text_phone'),
                 mode='w', encoding='UTF-8') as label_phone:
        # BASIC5000_0001 ~ BASIC5000_5000 까지 반복 실행
        for i in range(5000):
            # 발화ID
            filename = 'BASIC5000_%04d' % (i + 1)

            # 발화ID가 label_info에 포함되지 않을 경우 에러
            if not filename in label_info:
                print('Error: %s is not in %s' % (filename, original_label))
                exit()

            # 문자 라벨 정보
            chars = label_info[filename]['text_level2']
            # '、'와 '。'를 제거
            chars = chars.replace('、', '')
            chars = chars.replace('。', '')

            # 히라가나 라벨 정보
            kanas = label_info[filename]['kana_level3']
            # '、'를 제거
            kanas = kanas.replace('、', '')

            # 음소 라벨 정보
            phones = label_info[filename]['phone_level3']

            # 문자 라벨 파일에 한 글자씩 뛰어서 공백으로 구분
            # ('.join(list)은 리스트의 각 요소에 스페이스를 사이에 두고, 1문장으로 한다)
            label_char.write('%s %s\n' % (filename, ' '.join(chars)))

            # 히라가나 라벨 파일에 한 글자씩 뛰어서 공백으로 구분
            label_kana.write('%s %s\n' % (filename, ' '.join(kanas)))

            # 음소 라벨은 '-'를 스페이스로 치환해서 표시
            label_phone.write('%s %s\n' % (filename, phones.replace('-', ' ')))