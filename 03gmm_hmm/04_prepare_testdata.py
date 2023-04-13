# 04_prepare_testdata.py : HMM 테스트 데이터를 작성합니다.
#                          COUNTERSUFFIX26_01.wav를 단어마다 구분하고, 16kHz의 wav 데이터를 작성합니다.

import sox
import wave
import numpy as np
import os

if __name__ == "__main__":

    # 데이터는 COUNTERSUFFIX26_01을 사용한다
    original_wav = \
        '../data/original/jsut_ver1.1/' \
        'countersuffix26/wav/COUNTERSUFFIX26_01.wav'

    # 단어 정보
    word_info = [{'word': '一つ',
                  'phones': 'h i t o ts u',
                  'time': [0.17, 0.90]},
                 {'word': '二つ',
                  'phones': 'f u t a ts u',
                  'time': [1.23, 2.02]},
                 {'word': '三つ',
                  'phones': 'm i cl ts u',
                  'time': [2.38, 3.11]},
                 {'word': '四つ',
                  'phones': 'y o cl ts u',
                  'time': [3.42, 4.10]},
                 {'word': '五つ',
                  'phones': 'i ts u ts u',
                  'time': [4.45, 5.13]},
                 {'word': '六つ',
                  'phones': 'm u cl ts u',
                  'time': [5.52, 6.15]},
                 {'word': '七つ',
                  'phones': 'n a n a ts u',
                  'time': [6.48, 7.15]},
                 {'word': '八つ',
                  'phones': 'y a cl ts u',
                  'time': [7.52, 8.17]},
                 {'word': '九つ',
                  'phones': 'k o k o n o ts u',
                  'time': [8.51, 9.31]},
                 {'word': 'とお',
                  'phones': 't o o',
                  'time': [9.55, 10.10]}
                 ]

    # 음소 리스트
    phone_list_file = \
        './exp/data/train_small/phone_list'

    # 결과 출력 디렉토리
    out_dir = './exp/data/test'

    # 가공한 파형의 출력 디렉토리
    out_wav_dir = os.path.join(out_dir, 'wav')

    # 출력 디렉토리가 존재하지 않는 경우 작성
    os.makedirs(out_wav_dir, exist_ok=True)

    # sox에 의한 음성 변환 클래스를 호출
    tfm = sox.Transformer()
    # 샘플링 주파수를 16000Hz로 변환하도록 설정
    tfm.convert(samplerate=16000)

    downsampled_wav = os.path.join(out_wav_dir,
                                   os.path.basename(original_wav))

    # 파일이 존재하지 않는 경우 오류
    if not os.path.exists(original_wav):
        print('Error: Not found %s' % (original_wav))
        exit()

    # 샘플링 주파수 변환 및 저장을 실행
    tfm.build_file(input_filepath=original_wav,
                   output_filepath=downsampled_wav)

    # 다운샘플링한 음성을 불러오기
    with wave.open(downsampled_wav) as wav:
        # 샘플링 주파수
        sample_frequency = wav.getframerate()
        # wav 데이터 샘플 수
        num_samples = wav.getnframes()
        # wav 데이터 가져오기
        waveform = wav.readframes(num_samples)
        # 수치(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # wav 파일 리스트 파일
    wav_scp = os.path.join(out_dir, 'wav.scp')
    with open(wav_scp, mode='w') as scp_file:
        # 각 단어의 파형을 잘라내어 저장
        for n, info in enumerate(word_info):
            # 단어의 시간 정보를 얻기
            time = np.array(info['time'])
            # 시각[초]을 샘플 점으로 변환
            time = np.int64(time * sample_frequency)
            # 단어의 구간을 꺼내기
            cut_wav = waveform[time[0]: time[1]].copy()

            # 꺼낸 파형의 저장 파일 이름
            out_wav = os.path.join(out_wav_dir,
                                   "%d.wav" % (n + 1))
            # 꺼낸 파형을 저장
            with wave.open(out_wav, 'w') as wav:
                # 채널 수, 샘플 사이즈, 샘플링 주파수 설정
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_frequency)
                # 파형의 첫머리
                wav.writeframes(cut_wav)

            # wav 파일 목록에 써넣다
            scp_file.write('%d %s\n' %
                           ((n + 1), os.path.abspath(out_wav)))

    # 각 단어와 음소의 조합 목록(사전)을 작성
    lexicon = os.path.join(out_dir, 'lexicon.txt')
    with open(lexicon, mode='w') as f:
        for info in word_info:
            f.write('%s %s\n' \
                    % (info['word'], info['phones']))

    # 아래는 정답 라벨 작성 (음소 정렬 테스트에 사용)

    # 음소 리스트 파일을 열어 phone_list에 격납
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 음소 목록 파일에서 음소 가져오기
            phone = line.split()[0]
            # 음소 리스트의 뒤에 추가
            phone_list.append(phone)

    # 정답라벨리스트(음소는수치표기)작성
    label_file = os.path.join(out_dir, 'text_int')
    with open(label_file, mode='w') as f:
        for n, info in enumerate(word_info):
            label = info['phones'].split()
            # 양쪽 끝에 포즈 추가
            label.insert(0, phone_list[0])
            label.append(phone_list[0])
            # phone_list를 사용하여 음소를 수치로 변환하고 써넣기
            f.write('%d' % (n + 1))
            for ph in label:
                f.write(' %d' % (phone_list.index(ph)))
            f.write('\n')