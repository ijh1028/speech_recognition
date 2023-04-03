# DP매칭용 데이터를 작성합니다.
# 이 파일은 00_prepare/01prepare_wav.py 을 사용합니다

import sox
import os

if __name__ == "__main__":

    # wav파일이 전개된 디렉토리
    # 데이터는 repeat500을 사용
    original_wav_dir = '../data/original/jsut_ver1.1/repeat500/wav'

    # 포맷 변환한 wav파일을 출력하는 디렉토리
    out_wav_dir = './wav'

    # repeat500내에서 사용하는 세트 수
    num_set = 5

    # repeat500내에서 사용하는 1세트당의 언어 수
    num_utt_per_set = 10

    # 출력 디렉터리가 없는 경우 작성한다
    os.makedirs(out_wav_dir, exist_ok=True)

    # sox에 의한 음성 변환 클래스를 호출
    tfm = sox.Transformer()
    # 샘플링 주파수를 16000Hz로 변환하도록 설정
    tfm.convert(samplerate=16000)

    # 세트 x발화 수만큼 처리를 실행
    for set_id in range(num_set):
        for utt_id in range(num_utt_per_set):
            # wav파일 이름
            filename = 'REPEAT500_set%d_%03d' % (set_id + 1, utt_id + 1)
            # 변환 원래의 오리지날 데이터(48000Hz)의 파일명
            wav_path_in = os.path.join(original_wav_dir, filename + '.wav')
            # 형 변환 후의 데이터(16000Hz)의 보존 파일 이름
            wav_path_out = os.path.join(out_wav_dir, filename + '.wav')

            print(wav_path_in)
            # 파일이 존재하지 않으면 에러
            if not os.path.exists(wav_path_in):
                print('Error: Not found %s' % (wav_path_in))
                exit()

            # 샘플링 주파수 변환 및 저장을 실행
            tfm.build_file(input_filepath=wav_path_in,
                           output_filepath=wav_path_out)