# 05_compute_feat_test.py : MFCC 특징을 계산

import wave
import numpy as np
import os
import sys

class FeatureExtractor():
    ''' 특징량(FBANK, MFCC)을 추출하는 클래스
        sample_frequency: 입력 파형의 샘플링 주파수 [Hz]
        frame_length: 프레임 크기 [밀리초]
        frame_shift: 분석 간격(프레임 시프트) [밀리초]
        num_mel_bins: 멜 필터 뱅크의 수 (=FBANK 특징의 차원수)
        num_ceps: MFCC 특징의 차원수(0차원 포함)
        lifter_coef: 리프탈링 처리 파라미터
        low_frequency: 저주파수 대역 제거 컷오프 주파수 [Hz]
        high_frequency: 고주파수 대역 제거 컷오프 주파수 [Hz]
        dither: 디더링 처리 파라미터(잡음의 세기)
    '''

    # 클래스를 호출한 시점에서 처음에 한 번 실행되는 함수
    def __init__(self,
                 sample_frequency=16000,
                 frame_length=25,
                 frame_shift=10,
                 num_mel_bins=23,
                 num_ceps=13,
                 lifter_coef=22,
                 low_frequency=20,
                 high_frequency=8000,
                 dither=1.0):
        # 샘플링 주파수[Hz]
        self.sample_freq = sample_frequency
        # 프레임 사이즈르 밀리 초에서 샘플수로 변환
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        # 프레임 시프트를 밀리 초에서 샘플수로 변환
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        # 멜 필터 뱅크의 수
        self.num_mel_bins = num_mel_bins
        # MFCC의 차원수(0차 포함)
        self.num_ceps = num_ceps
        # 리프터 파라미터
        self.lifter_coef = lifter_coef
        # 저주파수 대역 제거 컷오프 주파수[Hz]
        self.low_frequency = low_frequency
        # 고주파수 대역 제거 컷오프 주파수[Hz]
        self.high_frequency = high_frequency
        # 디더링 계수
        self.dither_coef = dither

        # FFT의 포인트 수 = 프레임 사이즈가 2이상 이어야 함
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # 메일 필터 뱅크를 작성
        self.mel_filter_bank = self.MakeMelFilterBank()

        # 이산 코사인 변환(DCT)의 기저 행렬을 작성
        self.dct_matrix = self.MakeDCTMatrix()

        # 리프터(lifter)를 작성
        self.lifter = self.MakeLifter()

    def Herz2Mel(self, herz):
        ''' 주파수를 헤르츠에서 멜로 변환
        '''
        return (1127.0 * np.log(1.0 + herz / 700))

    def MakeMelFilterBank(self):
        ''' 메일 필터 뱅크를 작성
        '''
        # 멜축에서의 최대 주파수
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        # 멜축에서의 최소 주파수
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        # 최소에서 최대 주파수까지 멜축 상에서의 등간격 주파수를 얻기
        mel_points = np.linspace(mel_low_freq,
                                 mel_high_freq,
                                 self.num_mel_bins + 2)

        # 파워 스펙트럼의 차원수 = FFT 크기/2+1
        # ※Kaldi 구현에서는 나이키스트 주파수 성분(마지막 +1)은
        # 버리고 있지만, 본 실장에서는 버리지 않고 이용하고 있다.
        dim_spectrum = int(self.fft_size / 2) + 1

        # 멜 필터 뱅크(필터의 수 x 스펙트럼의 차원수)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 삼각 필터의 왼쪽 끝, 중앙, 오른쪽 끝의 멜 주파수
            left_mel = mel_points[m]
            center_mel = mel_points[m + 1]
            right_mel = mel_points[m + 2]
            # 파워 스펙트럼의 각 빈에 대응하는 무게를 계산
            for n in range(dim_spectrum):
                # 각 병에 대응하는 헤르츠축 주파수 계산
                freq = 1.0 * n * self.sample_freq / 2 / dim_spectrum
                # 멜 주파수로 변환
                mel = self.Herz2Mel(freq)
                # 그 병이 삼각 필터의 범위에 들어 있으면 무게를 계산
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filter_bank[m][n] = weight

        return mel_filter_bank

    def ExtractWindow(self, waveform, start_index, num_samples):
        '''
        한 프레임 분량의 파형 데이터를 추출하여 전처리를 실시,
        또한 로그 파워의 값도 계산
        '''
        # waveform에서 1프레임 분량의 파형을 추출
        window = waveform[start_index:start_index + self.frame_size].copy()

        # 디더링
        # (-dither_coef~dither_coef의 한결같은 난수 더하기)
        if self.dither_coef > 0:
            window = window \
                     + np.random.rand(self.frame_size) \
                     * (2 * self.dither_coef) - self.dither_coef

        # 직류 성분을 차단
        window = window - np.mean(window)

        # 이후의 처리를 실시하기 전에 파워를 요구
        power = np.sum(window ** 2)
        # 로그 계산 시에 -inf가 출력되지 않도록 플로어링 처리
        if power < 1E-10:
            power = 1E-10
        # 로그 설정
        log_power = np.log(power)

        # 프리엠퍼시스(고역강조)
        # # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window, np.array([1.0, -0.97]), mode='same')
        # numpy의 접힘에서는 0번째 요소가 처리되지 않는다
        # (window[i-1]가 존재하지 않으므로)
        # window[0-1]를 window[0]로 대용하여 처리
        window[0] -= 0.97 * window[0]

        # 윈도우에 hamming 함수 사용
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return window, log_power

    def ComputeFBANK(self, waveform):
        '''멜 필터 뱅크 특징(FBANK)을 계산하다
        출력 1: fbank_features: 멜 필터 뱅크 특징
        출력2: log_power: 로그 파워값(MFCC 추출시 사용)
        '''
        # 파형 데이터의 총 샘플 수
        num_samples = np.size(waveform)
        # 특징량의 총 프레임 수를 계산
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # 멜 필터 뱅크 특징
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        # 로그 파워(MFCC 특징을 구할 때 사용함)
        log_power = np.zeros(num_frames)

        # 한 프레임씩 특징량을 계산
        for frame in range(num_frames):
            # 분석의 개시 위치는 프레임 번호(0 시작)*프레임 시프트
            start_index = frame * self.frame_shift
            # 한 프레임 분량의 파형을 추출하여 전처리를 실시
            # 또한 로그 파워의 값도 얻는다
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)

            # 고속 푸리에 변환(FFT) 실행
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT 결과의 오른쪽 절반(음의 주파수 성분)을 제거하다
            # Kaldi의 실장에서는 나이키스트 주파수 성분(마지막 +1)은 버리고 있지만,
            # 본 실장에서는 버리지 않고 이용하고 있다
            spectrum = spectrum[:int(self.fft_size / 2) + 1]

            # 파워 스펙트럼을 계산
            spectrum = np.abs(spectrum) ** 2

            # 멜 필터 뱅크를 접어 넣다
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 로그 계산 시에 -inf가 출력되지 않도록 플로어링 처리를 한다.
            fbank[fbank < 0.1] = 0.1

            # 로그를 취해 fbank_features에 넣다
            fbank_features[frame] = np.log(fbank)

            # 로그 파워 값을 log_power에 더하다
            log_power[frame] = log_pow

        return fbank_features, log_power

    def MakeDCTMatrix(self):
        ''' 이산 코사인 변환(DCT)의 기저 행렬을 작성
        '''
        N = self.num_mel_bins
        # DCT 기저 행렬 (기저수(=MFCC의 차원수) x FBANK의 차원수)
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) \
                                * np.cos(((2.0 * np.arange(N) + 1) * k * np.pi) / (2 * N))

        return dct_matrix

    def MakeLifter(self):
        ''' 리프터를 계산
        '''
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter

    def ComputeMFCC(self, waveform):
        ''' MFCC를 계산
        '''
        # FBANK 및 로그 파워를 계산
        fbank, log_power = self.ComputeFBANK(waveform)

        # DCT의 기저 행렬과의 내적에 의해 DCT를 실시
        mfcc = np.dot(fbank, self.dct_matrix.T)

        # 리프터링
        mfcc *= self.lifter

        # MFCC의 0차원을 전처리를 하기 전 파형의 로그 파워로 대체
        mfcc[:, 0] = log_power

        return mfcc

if __name__ == "__main__":

    # 각 wav 파일 목록과 특징량 출력처
    test_wav_scp = './exp/data/test/wav.scp'
    test_out_dir = './exp/data/test/mfcc'

    # 샘플링 주파수[Hz]
    sample_frequency = 16000
    # 프레임 길이 [ms]
    frame_length = 25
    # 프레임 시프트[ms]
    frame_shift = 10
    # 저주파수 대역 제거 컷오프 주파수[Hz]
    low_frequency = 20
    # 고주파수 대역 제거 컷오프 주파수[Hz]
    high_frequency = sample_frequency / 2
    # 멜 필터 뱅크의 수
    num_mel_bins = 23
    # MFCC의 차원수
    num_ceps = 13
    # 디더링 계수
    dither = 1.0

    # 난수 시드 설정(디더링 처리 결과 재현성 담보)
    np.random.seed(seed=0)

    # 특징량 추출 클래스를 호출
    feat_extractor = FeatureExtractor(
        sample_frequency=sample_frequency,
        frame_length=frame_length,
        frame_shift=frame_shift,
        num_mel_bins=num_mel_bins,
        num_ceps=num_ceps,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        dither=dither)

    # wav 파일 리스트와 출력처를 리스트로 만들기
    wav_scp_list = [test_wav_scp]
    out_dir_list = [test_out_dir]

    # 각 세트에 대해 처리를 실행
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print('Input wav_scp: %s' % (wav_scp))
        print('Output directory: %s' % (out_dir))

        # 특징량 파일의 경로, 프레임 수,
        # 차원수를 적은 목록
        feat_scp = os.path.join(out_dir, 'feats.scp')

        # 출력 디렉토리가 존재하지 않는 경우 작성
        os.makedirs(out_dir, exist_ok=True)

        # wav목록읽기모드,
        # 특징량 목록을 쓰기 모드로 열기
        with open(wav_scp, mode='r') as file_wav, \
                open(feat_scp, mode='w') as file_feat:
            # wav 목록을 한 줄씩 불러들이다
            for line in file_wav:
                # 각 행에는 발화 ID와 wav 파일의 패스가스페이스 구분으로 기재되어 있으므로
                # split 함수를 사용하여 칸막이 행을 리스트형 변수로 변환
                parts = line.split()
                # 0번째가 발화ID
                utterance_id = parts[0]
                # 첫번째가 wav 파일 경로
                wav_path = parts[1]

                # wav 파일을 읽어들여 특징량을 계산
                with wave.open(wav_path) as wav:
                    # 샘플링 주파수 검사
                    if wav.getframerate() != sample_frequency:
                        sys.stderr.write('The expected \
                            sampling rate is 16000.\n')
                        exit(1)
                    # wav 파일이 1채널(모노)
                    # 데이터임을 체크
                    if wav.getnchannels() != 1:
                        sys.stderr.write('This program \
                            supports monaural wav file only.\n')
                        exit(1)

                    # wav 데이터 샘플 수
                    num_samples = wav.getnframes()

                    # wav 데이터 가져오기
                    waveform = wav.readframes(num_samples)

                    # 가져온 데이터는 이진 값
                    # (16bit integer)이므로 수치(정수)로 변환
                    waveform = np.frombuffer(waveform, dtype=np.int16)

                    # MFCC를 계산
                    mfcc = feat_extractor.ComputeMFCC(waveform)

                # 특징량의 프레임 수와 차원 수 취득
                (num_frames, num_dims) = np.shape(mfcc)

                # 특징량 파일 이름(splitext에서 확장자 제거)
                out_file = os.path.splitext(os.path.basename(wav_path))[0]
                out_file = os.path.join(os.path.abspath(out_dir),
                                        out_file + '.bin')

                # 데이터를 float32 형식으로 변환
                mfcc = mfcc.astype(np.float32)

                # 데이터를 파일로 출력
                mfcc.tofile(out_file)
                # 발화 ID, 특징량 파일의 패스, 프레임 수,
                # 차원수를 특징량 리스트에 써 넣다
                file_feat.write("%s %s %d %d\n" %
                                (utterance_id, out_file, num_frames, num_dims))