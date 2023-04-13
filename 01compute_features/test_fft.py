# test_fft.py : wav 파일의 특정 시각 파형을 푸리에 변환하여 로그 진폭 스펙트럼을 표시

import wave
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # wav 파일 열기
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # 분석 시각. BASIC5000_0001.wav에서는 아래 시각에 음소 '0'을 발화
    target_time = 0.58

    # FFT(고속 푸리에 변화)한 범위의 샘플 수는 2의 제곱수여야 한다
    fft_size = 1024

    # 시각화 결과 파일(png파일)
    out_plot = './spectrum.png'

    # wav 파일을 열고 아래 코드들을 수행
    with wave.open(wav_file) as wav:
        # 샘플링 주파수 [Hz] 확인
        sampling_frequency = wav.getframerate()
        # wav 데이터 읽기
        waveform = wav.readframes(wav.getnframes())
        # 읽어온 데이터는 바이너리 값(16bit integer)이므로 수치(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # 분석 시각을 샘플 번호로 변환
    target_index = np.int(target_time * sampling_frequency)
    # FFT를 실행하는 구간만큼의 파형 데이터를 도출
    frame = waveform[target_index: target_index + fft_size]
    # FFT 적용
    spectrum = np.fft.fft(frame)
    # 진폭 스펙트럼 확인
    absolute = np.abs(spectrum)
    # 진폭 스펙트럼은 좌우 대칭이므로 좌측 반만 이용
    absolute = absolute[:np.int(fft_size / 2) + 1]
    # 로그 함수를 취하고 로그 진폭 스펙트럼 계산
    log_absolute = np.log(absolute + 1E-7)

    # 시간 파형과 로그 진폭 스펙트럼을 시각화
    plt.figure(figsize=(10, 10))
    # 그림 영역을 종으로 2분활하여 위쪽에 시간 파형 표시
    plt.subplot(2, 1, 1)

    # 횡축(시간축) 생성
    time_axis = target_time \
                + np.arange(fft_size) / sampling_frequency

    # 파형 그리기
    plt.plot(time_axis, frame)

    # 시각화한 그림의 제목과 횡축, 종축 라벨 정의
    plt.title('waveform')
    plt.xlabel('Time [sec]')
    plt.ylabel('Value')

    # 횡축의 표시 영역을 분석 구간의 시각으로 제한
    plt.xlim([target_time,
              target_time + fft_size / sampling_frequency])

    # 2분할한 그림 영역 밑에 로그 진폭 스펙트럼을 표시
    plt.subplot(2, 1, 2)
    # 횡축(주파수 축) 생성
    freq_axis = np.arange(np.int(fft_size / 2) + 1) \
                * sampling_frequency / fft_size
    # 로그 진폭 스펙트럼 시각화
    plt.plot(freq_axis, log_absolute)
    # 시각화한 그림의 제목과 횡축, 종축 라벨 정의
    plt.title('log-absolute spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')
    # 횡축 표시 영역을 0 ~ 최대 주파수로 제한
    plt.xlim([0, sampling_frequency / 2])

    # 시각화된 결과물 저장
    plt.savefig('out_plot')