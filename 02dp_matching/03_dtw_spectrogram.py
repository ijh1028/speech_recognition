# 03_dtw_spectrogram.py :  음성 스펙트로그램을 작성한 후, 얼라이먼트 정보를 이용하여 스펙트로그램을 늘린다

import wave
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # wav파일 열기
    wav_file_1 = './wav/REPEAT500_set1_009.wav'
    wav_file_2 = './wav/REPEAT500_set2_009.wav'

    alignment_file = './alignment.txt'

    # 샘플링 주파수
    sample_frequency = 16000
    # 프레임 사이즈[ms]
    frame_size = 25
    # 프레임 시프트[ms]
    frame_shift = 10

    # 플롯을 출력하는 파일(png 파일)
    out_plot = './dtw_spectrogram.png'

    # 프레임 사이즈를 밀리 초에서 샘플 수에 변환
    frame_size = int(sample_frequency * frame_size * 0.001)

    # 프레임 시프트를 밀리 초에서 샘플 수로 변환
    frame_shift = int(sample_frequency * frame_shift * 0.001)

    # FFT를 실시하는 범위의 표본 수를,
    # 프레임 사이즈 이상의 2의 멱승에 설정
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    # 배열 정보를 얻기
    alignment = []
    with open(alignment_file, mode='r') as f:
        for line in f:
            parts = line.split()
            alignment.append([int(parts[0]), int(parts[1])])

    # 플롯 그리기 영역 생성
    plt.figure(figsize=(10, 10))

    # 2개의 wavfile에 대해서 이하를 실행
    for file_id, wav_file in enumerate([wav_file_1, wav_file_2]):
        # wav파일을 열고 이후의 처리
        with wave.open(wav_file) as wav:
            # wav데이터 정보를 읽기
            num_samples = wav.getnframes()
            waveform = wav.readframes(num_samples)
            waveform = np.frombuffer(waveform, dtype=np.int16)

        # 단시간 푸리에 변환을 했을 때 총 프레임 수를 계산
        num_frames = (num_samples - frame_size) // frame_shift + 1

        # 스펙트로그램의 행렬을 준비
        spectrogram = np.zeros((num_frames, fft_size))

        # 1프레임씩 진폭 스펙트럼을 계산하다
        for frame_idx in range(num_frames):
            # 분석의 시작 위치는, 프레임 번호(0시작) * 프레임 시프트
            start_index = frame_idx * frame_shift

            # 1프레임 분의 파형을 추출
            frame = waveform[start_index: \
                             start_index + frame_size].copy()

            # 해밍 함수
            frame = frame * np.hamming(frame_size)

            # 고속 푸리에 변환(FFT)을 실행
            spectrum = np.fft.fft(frame, n=fft_size)

            # 로그 진폭 스펙트럼을 계산
            log_absolute = np.log(np.abs(spectrum) + 1E-7)

            # 계산 결과를 스펙트로그램에 저장
            spectrogram[frame_idx, :] = log_absolute

        # 스펙트 로그램을 얼라인먼트에 맞추어 확대
        dtw_spectrogram = np.zeros((len(alignment), fft_size))
        for t in range(len(alignment)):
            # 대응하는 프레임 번호
            idx = alignment[t][file_id]
            # 대응하는 프레임 번호의 스펙트로그램을 복사하고 늘리기
            dtw_spectrogram[t, :] = spectrogram[idx, :]

        # 시간 파형과 스펙트로그램을 플롯 그리기 영역을 세로로 2분할하여 스펙트로그램을 플롯하다
        plt.subplot(2, 1, file_id + 1)

        # 스펙트 로그램의 최대치를 0에 맞추어 컬러 맵의 범위를 조정
        dtw_spectrogram -= np.max(dtw_spectrogram)
        vmax = np.abs(np.min(dtw_spectrogram)) * 0.0
        vmin = - np.abs(np.min(dtw_spectrogram)) * 0.7

        # 히스토그램을 플롯
        plt.imshow(dtw_spectrogram.T[-1::-1, :],
                   extent=[0, len(alignment) * \
                           frame_shift / sample_frequency,
                           0, sample_frequency],
                   cmap='gray',
                   vmax=vmax,
                   vmin=vmin,
                   aspect='auto')
        plt.ylim([0, sample_frequency / 2])

        # 플롯 제목과 가로축과 세로축 라벨 정의
        plt.title('spectrogram')
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')

    # 플롯을 저장
    plt.savefig(out_plot)