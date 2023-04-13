# wav 파일의 일부 구간을 푸리에 변환하고,
# 진폭 스펙트럼을 플롯합니다.

import wave
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # wav 파일 열기
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # 분석하는 시각. BASIC5000_0001.wav에서는
    # 이하의 시각은 음소 "o"를 발화하고 있다.
    target_time = 0.58

    # FFT(고속 푸리에 변환)를 수행하는 범위의 샘플 수
    # 2가 되어야 할 곱일 필요가 있다
    fft_size = 1024

    # 플롯을 출력할 파일(png파일)
    out_plot = './pre_emphasis.png'

    # wav 파일을 열고 이후에 처리
    with wave.open(wav_file) as wav:
        # 샘플링 주파수 [Hz]
        sampling_frequency = wav.getframerate()

        # wav 데이터 가져오기
        waveform = wav.readframes(wav.getnframes())

        # 가져온 데이터는 이진값(16bit integer)그래서 수치(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # 분석하는 시각을 샘플 번호로 변환
    target_index = np.int(target_time * sampling_frequency)

    # FFT를 실시하는 구간 만큼의 파형 데이터를 골라내기
    frame = waveform[target_index: target_index + fft_size]

    frame_emp = np.convolve(frame, np.array([1.0, -0.97]), mode='same')
    # numpy의 상승에서는 0번째 요소가 처리되지 않는다(window[i-1]이 존재하지 않기 때문에)때문에
    # window[0-1]을 window[0]로 대용하고 처리
    frame_emp[0] -= 0.97 * frame_emp[0]

    h = np.zeros(fft_size)
    h[0] = 1.0
    h[1] = -0.97

    frame = frame * np.hamming(fft_size)
    frame_emp = frame_emp * np.hamming(fft_size)

    # FFT를 실시
    spectrum = np.fft.fft(frame)
    spectrum_emp = np.fft.fft(frame_emp)
    spectrum_h = np.fft.fft(h)

    # 진폭 스펙트럼을 얻기
    absolute = np.abs(spectrum)
    absolute_emp = np.abs(spectrum_emp)
    absolute_h = np.abs(spectrum_h)

    # 진폭 스펙트럼은 좌우 대칭이므로, 왼쪽 절반까지만을 이용
    absolute = absolute[:np.int(fft_size / 2) + 1]
    absolute_emp = absolute_emp[:np.int(fft_size / 2) + 1]
    absolute_h = absolute_h[:np.int(fft_size / 2) + 1]

    # 대수를 취하고 로그 진폭 스펙트럼을 계산
    log_absolute = np.log(absolute + 1E-7)
    log_absolute_emp = np.log(absolute_emp + 1E-7)
    log_absolute_h = np.log(absolute_h + 1E-7)

    # 시간 파형과 로그 진폭 스펙트럼을 플롯

    # 플롯의 묘화 영역을 작성
    plt.figure(figsize=(10, 10))

    # 2분할된 그리기 영역 아래쪽에
    # 로그 진폭 스펙트럼을 플롯
    plt.subplot(3, 1, 1)

    # 횡축(주파수 축)를 작성
    freq_axis = np.arange(np.int(fft_size / 2) + 1) \
                * sampling_frequency / fft_size

    # 로그 진폭 스펙트럼을 플롯
    plt.plot(freq_axis, log_absolute, color='k')

    # 플롯의 제목과, 가로 축과 세로축의 라벨을 정의
    # plt.title('log-absolute spectrum without pre-emphasis (x)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 횡축의 표시 영역을 0~최대 주파수에 제한
    plt.xlim([0, sampling_frequency / 2])
    plt.ylim([0, 15])

    # 2분할된 선묘화 영역의 아래쪽에
    # 로그 진폭 스펙트럼을 플롯
    plt.subplot(3, 1, 2)

    # 횡축(주파수 축)를 작성
    freq_axis = np.arange(np.int(fft_size / 2) + 1) \
                * sampling_frequency / fft_size

    # 로그 진폭 스펙트럼을 플롯
    plt.plot(freq_axis, log_absolute_emp, color='k')

    # 플롯의 제목과, 가로 축과 세로축의 라벨을 정의
    # plt.title('log-absolute spectrum with pre-emphasis (x_emp)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 가로축 표시 영역을 0~최대 주파수로 제한
    plt.xlim([0, sampling_frequency / 2])
    plt.ylim([0, 15])

    plt.subplot(3, 1, 3)

    # 가로축(주파수축)을 작성
    freq_axis = np.arange(np.int(fft_size / 2) + 1) \
                * sampling_frequency / fft_size

    # 로그 진폭 스펙트럼을 플롯
    plt.plot(freq_axis, log_absolute_emp - log_absolute, linestyle='dashed', color='k')
    plt.plot(freq_axis, log_absolute_h, color='k')

    # 플롯 제목과 가로축과 세로축 라벨 정의
    # plt.title('log-absolute spectra of pre-emphasis filter (h) and (x_emp - x)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 횡축의 표시 영역을 0~최대 주파수에 제한
    plt.xlim([0, sampling_frequency / 2])

    # 플롯을 저장
    plt.savefig(out_plot)