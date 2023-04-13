import numpy as np
import matplotlib.pyplot as plt

# 주파수를 헤르츠에서 멜로 변환
def Herz2Mel(herz):
    return (1127.0 * np.log(1.0 + herz / 700))

if __name__ == "__main__":

    # 최대 주파수[Hz]
    max_herz = 8000

    # FFT의 포인트 수
    fft_size = 4096

    # 필터 뱅크의 수
    num_mel_bins = 7

    # 플롯을 출력할 파일(png파일)
    out_plot = './mel_bank.png'

    # mel축에서 최대 주파수
    max_mel = Herz2Mel(max_herz)

    # mel축 위에서 등간격 주파수를 얻기
    mel_points = np.linspace(0, max_mel, num_mel_bins + 2)

    # 파워 스펙트럼의 차원 수=FFT크기/2+1
    dim_spectrum = int(fft_size / 2) + 1

    # mel필터 뱅크(필터의 수 x스펙트럼의 차원 수)
    mel_filter_bank = np.zeros((num_mel_bins, dim_spectrum))
    for m in range(num_mel_bins):
        # 삼각 필터의 왼쪽 끝, 중앙, 우단의 메일 주파수
        left_mel = mel_points[m]
        center_mel = mel_points[m + 1]
        right_mel = mel_points[m + 2]
        # 파워 스펙트럼의 각 빈에 대응하는 무게를 계산하다
        for n in range(dim_spectrum):
            # 각 병에 대응하는 헤르츠의 주파수를 계산
            freq = 1.0 * n * max_herz / dim_spectrum
            # 멜 주파수로 변환
            mel = Herz2Mel(freq)
            # 그 병이 삼각 필터의 범위에 들어 있으면 무게를 계산
            if mel > left_mel and mel < right_mel:
                if mel <= center_mel:
                    weight = (mel - left_mel) / (center_mel - left_mel)
                else:
                    weight = (right_mel - mel) / (right_mel - center_mel)
                mel_filter_bank[m][n] = weight

    # 플롯 그리기 영역 생성
    plt.figure(figsize=(6, 4))

    # 가로축(주파수축)을 작성하다
    freq_axis = np.arange(dim_spectrum) \
                * max_herz / dim_spectrum

    for m in range(num_mel_bins):
        # 필터 뱅크를 줄거리
        plt.plot(freq_axis, mel_filter_bank[m], color='k')
        # plt.plot(freq_axis, mel_filter_bank[m])

    plt.xlabel('Frequency [Hz]')

    # 가로축 표시 영역을 0~최대 주파수로 제한
    plt.xlim([0, max_herz])

    # 플롯을 저장
    plt.savefig(out_plot)