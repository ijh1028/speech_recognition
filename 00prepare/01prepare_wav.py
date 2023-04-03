# 다운로드한 wav 파일을 샘플링 레이트 16000Hz의 데이터로 변환
# 변환한 wav 데이터의 리스트를 작성

# 샘플링 주파수를 변환하기 위해 sox사용
import sox

# os
import os

# 메인 함수
if __name__ == "__main__":

    # wav 파일이 전개된 디렉토리
    original_wav_dir = '../data/original/jsut_ver1.1/basic5000/wav'

    # 포맷 변환한 wav파일을 출력하는 디렉토리
    out_wav_dir = '../data/wav'

    # wav 데이터 라벨을 저장하는 디렉토리
    out_scp_dir = '../data/label/all'

    # 출력 디렉토리가 없는 경우 작성
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(out_scp_dir, exist_ok=True)

    # sox에 의한 음성 변환 클래스를 호출
    tfm = sox.Transformer()
    # 샘플링 주파수를 16000Hz로 변환하도록 설정
    tfm.convert(samplerate=16000)

    # wav 데이터의 리스트 파일을 기입 모드로 열고, 이후의 처리 실시
    with open(os.path.join(out_scp_dir, 'wav.scp'), mode='w') as scp_file:
        # BASIC5000_0001.wav ~ BASIC5000_5000.wav에 대해 처리 반복 실행
        for i in range(5000):
            filename = 'BASIC5000_%04d' % (i + 1)
            # 변환원의 원본 데이터(48000Hz)의 파일명
            wav_path_in = os.path.join(original_wav_dir, filename + '.wav')
            # 변환 후 데이터(16000Hz) 저장 파일명
            wav_path_out = os.path.join(out_wav_dir, filename + '.wav')

            print(wav_path_in)
            # 파일이 존재하지 않는 경우 오류
            if not os.path.exists(wav_path_in):
                print('Error: Not found %s' % (wav_path_in))
                exit()

            # 샘플링 주파수 변환 및 저장을 실행
            tfm.build_file(input_filepath=wav_path_in,
                           output_filepath=wav_path_out)

            # wav 파일 스크립트
            scp_file.write('%s %s\n' %
                           (filename, os.path.abspath(wav_path_out)))