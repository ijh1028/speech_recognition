import wave
import numpy as np
import os
import sys

class FeatureExtractor():
    ''' 특징값(FBANK, MFCC)을 추출하는 클래스
    sample_frequency : 입력 파형 샘플링 주파수[Hz]
    frame_size : 프레임 사이즈[msec]
    frame_shift : 분석 간격(프레임 시프트)[msec]
    num_mel_bins : Mel 필터 뱅크 수 (=FBANK 특징 차원 수)
    num_ceps : MFCC 특징 차원 수(0차원 포함)
    lifter_coef : 리프터링 처리 매개변수
    low_frequency : 저주파수 대역 제거 cut-off 주파수[Hz]
    high_frequency : 고주파수 대역 제거 cut-off 주파수[Hz]
    dither : 디더링 처리 매개변수(잡은 크기)
    '''

    # 클래스를 불러온 시점에서 최초 1회 실행되는 함수
    def __init__(self, sample_frequency=16000,
                 frame_size=25, frame_shift=10,
                 num_mel_bins=23, num_ceps=13,
                 lifter_coef=22, low_frequence=20,
                 high_frequence=8000, dither=1.0):
        # 샘플링 주파수[Hz]
        self.sample_freq = sample_frequency
        # 프레임 사이즈를 msec에서 샘플 수로 변환
        self.frame_size = int(sample_frequency * frame_size * 0.001)
        # 프레임 시프트를 msec에서 샘플 수로 변환
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        # Mel 필터 뱅크 수
        self.num_mel_bins = num_mel_bins
        # MFCC 차원 수(0차 포함)
        self.num_ceps = num_ceps
        # 리프터링 매개변수
        self.lifter_coef = lifter_coef
        # 저주파수 대역 제거 절단 주파수[Hz]
        self.low_frequence = low_frequence
        # 고주파수 대역 제거 절단 주파수[Hz]
        self.high_frequence = high_frequence
        # 디더링 개수
        self.dither_coef = dither

        # FFT 포인트 수 = 프레임 사이즈 이상의 2제곱
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # Mel 필터 뱅크를 작성
        self.mel_filter_bank = self.MakeMelFilterBank()

    def Herz2Mel(self, herz):
        ''' 주파수를 헤르츠에서 Mel로 변환한다
        '''
        return (1127.0 * np.log(1.0 + herz / 700))
    
    def MakeMelFilterBank(self):
        ''' Mel 필터 뱅크를 생성한다
        '''
        # Mel 축에서 최대 주파수
        mel_high_freq = self.Herz2Mel(self.high_frequence)
        # Meql 축에서 최소 주파수
        mel_low_freq = self.Herz2Mel(self.low_frequence)
        # 최소에서 최대 주파수까지 Mel 축 위에서 동일 간격으로 주파수
        mel_points = np.linspace(mel_low_freq, mel_high_freq, self.num_mel_bins+2)
        # 파워 스펙트럼 차원 수 = FFT사이즈/ 2 + 1
        dim_spectrum = int(self.fft_size / 2) + 1

        # 멜 필터 뱅크(필터 수 * 스펙트럼 차원 수)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 삼각 필터 좌측 끝 중앙, 우측 끝 멜 주파수
            left_mel = mel_points[m]
            center_mel = mel_points[m+1]
            right_mel = mel_points[m+2]
            # 파워 스펙트럼의 각 bin에 대응하는 가중치를 계산
            for n in range(dim_spectrum):
                # 각 bin에 대응하는 Hz축 주파수의 계산
                freq = 1.0 * n * self.sample_freq/2 / dim_spectrum
                # Mel 주파수로 변환
                mel = self.Herz2Mel(freq)
                # 그 bin이 삼각 필터 범위에 있다면 가중치를 계산
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (mel - left-mel) / (right_mel-center_mel)
                    mel_filter_bank[m][n] = weight

        return mel_filter_bank
    
    def ExtractWindow(self, waveform, start_index, num_samples):
        '''
        1 프레임 분량의 파형 데이터를 추출하여 전처리하고 로그 파워값을 계산
        '''
        # waveform에서 1 프레임 분량의 파형을 추출
        window = waveform[start_index:start_index + self.frame_size].copy()

        # 디더링
        # (-dither_coef～dither_coef사이 값을 난수로 추가)
        if self.dither_coef > 0:
            window = window \
                     + np.random.rand(self.frame_size) \
                     * (2*self.dither_coef) - self.dither_coef

        # 직류 성분을 제거
        window = window - np.mean(window)

        # 아래 처리를 실행하기 전에 파워 구하기
        power = np.sum(window ** 2)
        # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
        if power < 1E-10:
            power = 1E-10
        # 로그 취하기
        log_power = np.log(power)

        # Pre Emphasis(고역 강조)
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window,np.array([1.0, -0.97]), mode='same')
        # numpy.convolve는 0번째 요소가 처리되지 않기
        # (window[i-1]가 없기) 떄문에 window[0-1]을 window[0]로 대체하여 처리
        window[0] -= 0.97*window[0]

        # 해밍 창 함수를 적용
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return window, log_power


    def ComputeFBANK(self, waveform):
        '''로그 Mel 필터 뱅크 특성(FBANK)을 계산
        출력1: fbank_features: 로그 Mel 필터 뱅크 특징
        출력2: log_power: 로그 파워 값(MFCC 추출 시에 사용)
        '''
        # 파형 데이터 총 샘플 수
        num_samples = np.size(waveform)
        # 특징값의 총 프레임 수를 계산
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # Mel 필터 뱅크 특징
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        # 로그 파워(MFCC 특성을 구할 때 사용)
        log_power = np.zeros(num_frames)

        # 1 프레임마다 특징값을 계산
        for frame in range(num_frames):
            # 분석 시작 위치는 프레임 번호(0에서 시작) * 프레임 시프트
            start_index = frame * self.frame_shift
            # 1 프레임 분량의 파형을 추출하여 전처리를 수행하고 로그 파워 값도 얻기
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)
            
            # 파워 스펙트럼 계산
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT 결과의 오른쪽 절반(음의 주파수 성분)을 제거
            spectrum = spectrum[:int(self.fft_size/2) + 1]
            spectrum = np.abs(spectrum) ** 2

            # Mel 필터 뱅크 계산
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
            fbank[fbank<0.1] = 0.1

            # 로그를 취해서 fbank_features에 첨가
            fbank_features[frame] = np.log(fbank)

            # 로그 파워 값을 log_power에 첨가
            log_power[frame] = log_pow

        return fbank_features, log_power

if __name__ == "__main__":
    
    # 각 wav 파일 목록과 특징값 출력 위치
    train_small_wav_scp = '../data/label/train_small/wav.scp'
    train_small_out_dir = './fbank/train_small'
    train_large_wav_scp = '../data/label/train_large/wav.scp'
    train_large_out_dir = './fbank/train_large'
    dev_wav_scp = '../data/label/dev/wav.scp'
    dev_out_dir = './fbank/dev'
    test_wav_scp = '../data/label/test/wav.scp'
    test_out_dir = './fbank/test'

    # 샘플링 주파수 [Hz]
    sample_frequency = 16000
    # 샘플링 길이 [msec]
    frame_length = 25
    # 프레임 시프트 [msec]
    frame_shift = 10
    # 저주파수 대역 제거 절단 주파수 [Hz]
    low_frequency = 20
    # 고주파수 대역 제거 절단 주파수 [Hz]
    high_frequency = sample_frequency / 2
    # 로그 Mel 필터 뱅크 특징 차원 수
    num_mel_bins = 40
    # 디더링 개수
    dither=1.0

    # 특징값 추출 클래스 불러오기
    feat_extractor = FeatureExtractor(
                       sample_frequency=sample_frequency, 
                       frame_length=frame_length, 
                       frame_shift=frame_shift, 
                       num_mel_bins=num_mel_bins, 
                       low_frequency=low_frequency, 
                       high_frequency=high_frequency, 
                       dither=dither)

    # wav 파일 목록과 출력 위치를 리스트로 생성
    wav_scp_list = [train_small_wav_scp, 
                    train_large_wav_scp, 
                    dev_wav_scp, 
                    test_wav_scp]
    out_dir_list = [train_small_out_dir, 
                    train_large_out_dir, 
                    dev_out_dir, 
                    test_out_dir]

    # 각 세트에 대한 처리를 실행
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print('Input wav_scp: %s' % (wav_scp))
        print('Output directory: %s' % (out_dir))

        # 특징값 파일 경로, 프레임 수, 차원 수를 기록한 리스트
        feat_scp = os.path.join(out_dir, 'feats.scp')