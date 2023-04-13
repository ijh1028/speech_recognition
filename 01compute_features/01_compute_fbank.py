import wave
import numpy as np
import os
import sys


class FeatureExtractor():
    '''
    특징값(FBANK, MFCC)을 추출하는 클래스
    sample_frequency : 입력 파형 샘플링 주파수[Hz]
    frame_length : 프레임 사이즈[msec]
    frame_shift : 분석 간격(프레임 시프트)[msec]
    num_mel_bins : Mel 필터 뱅크 수 (=FBANK 특징 차원 수)
    num_ceps : MFCC 특징 차원 수(0차원 포함)
    lifter_coef : 리프터링 처리 매개변수
    low_frequency : 저주파수 대역 제거 cut-off 주파수[Hz]
    high_frequency : 고주파수 대역 제거 cut-off 주파수[Hz]
    dither : 디더링 처리 매개변수(잡은 크기)
    '''

    # 클래스를 불러온 시점에서 최초 1회 실행되는 함수
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
        # 프레임 사이즈를 msec에서 샘플 수로 변환
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        # 프레임 시프트를 msec에서 샘플 수로 변환
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        # Mel 필터 뱅크 수
        self.num_mel_bins = num_mel_bins
        # MFCC 차원 수(0차 포함)
        self.num_ceps = num_ceps
        # 리프터링 매개변수
        self.lifter_coef = lifter_coef
        # 저주파수 대역 제거 절단 주파수[Hz]
        self.low_frequency = low_frequency
        # 고주파수 대역 제거 절단 주파수[Hz]
        self.high_frequency = high_frequency
        # 디더링 개수
        self.dither_coef = dither

        # FFT 포인트 수 = 프레임 사이즈 이상의 2제곱
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # Mel 필터 뱅크를 작성
        self.mel_filter_bank = self.MakeMelFilterBank()

        # 이산 코사인 변환(DCT)의 기저 행렬을 작성
        self.dct_matrix = self.MakeDCTMatrix()

        # 리프터(lifter)를 작성
        self.lifter = self.MakeLifter()

    def Herz2Mel(self, herz):
        ''' 주파수를 헤르츠에서 Mel로 변환한다
        '''
        return (1127.0 * np.log(1.0 + herz / 700))

    def MakeMelFilterBank(self):
        ''' Mel 필터 뱅크를 생성한다
        '''
        # Mel 축에서 최대 주파수
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        # Meql 축에서 최소 주파수
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        # 최소에서 최대 주파수까지 Mel 축 위에서 동일 간격으로 주파수
        mel_points = np.linspace(mel_low_freq,
                                 mel_high_freq,
                                 self.num_mel_bins + 2)
        # 파워 스펙트럼 차원 수 = FFT사이즈/ 2 + 1
        dim_spectrum = int(self.fft_size / 2) + 1

        # 멜 필터 뱅크(필터 수 * 스펙트럼 차원 수)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 삼각 필터 좌측 끝 중앙, 우측 끝 멜 주파수
            left_mel = mel_points[m]
            center_mel = mel_points[m + 1]
            right_mel = mel_points[m + 2]
            # 파워 스펙트럼의 각 bin에 대응하는 가중치를 계산
            for n in range(dim_spectrum):
                # 각 bin에 대응하는 Hz축 주파수의 계산
                freq = 1.0 * n * self.sample_freq / 2 / dim_spectrum
                # Mel 주파수로 변환
                mel = self.Herz2Mel(freq)
                # 그 bin이 삼각 필터 범위에 있다면 가중치를 계산
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
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
                     * (2 * self.dither_coef) - self.dither_coef

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
        window = np.convolve(window, np.array([1.0, -0.97]), mode='same')
        # nnumpy.convolve는 0번째 요소가 처리되지 않기
        # (window[i-1]가 없기) 떄문에 window[0-1]을 window[0]로 대체하여 처리，
        window[0] -= 0.97 * window[0]

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
            # 11 프레임 분량의 파형을 추출하여 전처리를 수행하고 로그 파워 값도 얻기
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)

            # 파워 스펙트럼 계산
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT 결과의 오른쪽 절반(음의 주파수 성분)을 제거
            spectrum = spectrum[:int(self.fft_size / 2) + 1]
            spectrum = np.abs(spectrum) ** 2

            # Mel 필터 뱅크 계산
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
            fbank[fbank < 0.1] = 0.1

            # 로그를 취해서 fbank_features에 첨가
            fbank_features[frame] = np.log(fbank)

            # 로그 파워 값을 log_power에 첨가
            log_power[frame] = log_pow

        return fbank_features, log_power

    def MakeDCTMatrix(self):
        ''' 이산 코사인 변환(DCT)의 기저 행렬을 작성
        '''
        N = self.num_mel_bins
        # DCT 기저 행렬 (기저수 (=MFCC의 차원수) x FBANK의 차원수)
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) \
                                * np.cos(((2.0 * np.arange(N) + 1) * k * np.pi) / (2 * N))

        return dct_matrix

    def MakeLifter(self):
        ''' 리프터 계산
        '''
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter

    def ComputeMFCC(self, waveform):
        ''' MFCC 계산
        '''
        # FBANK 및 로그 파워를 계산
        fbank, log_power = self.ComputeFBANK(waveform)

        # DCT의 기저 행렬과의 내적에 의해 DCT를 실시
        mfcc = np.dot(fbank, self.dct_matrix.T)

        # 리프터 계산
        mfcc *= self.lifter

        # MFCC의 0차원을 전처리를 하기 전 파형의 로그 파워로 대체
        mfcc[:, 0] = log_power

        return mfcc


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
    dither = 1.0

    # 난수 시드 설정(디더링 처리 결과 재현성 담보)
    np.random.seed(seed=0)

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

        # 출력 디렉터리가 없는 경우 작성
        os.makedirs(out_dir, exist_ok=True)

        # wav 목록 읽기 모드
        # 특징값 리스트를 쓰기 모드로 열기
        with open(wav_scp, mode='r') as file_wav, \
                open(feat_scp, mode='w') as file_feat:
            # wav 목록을 1행씩 읽기
            for line in file_wav:
                # 각 행에는 발화 ID와 wav 파일 경로가 스페이스로 구분되어 있음
                # split 함수를 써서 스페이스 구분 행을 리스트형 변수로 변환
                parts = line.split()
                # 0번째가 발화 ID
                utterance_id = parts[0]
                # 1번째가 wav 파일 경로
                wav_path = parts[1]

                # wav 파일을 읽고 특징값을 계산
                with wave.open(wav_path) as wav:
                    # 샘플링 주파수 검사
                    if wav.getframerate() != sample_frequency:
                        sys.stderr.write('The expected \
                            sampling rate is 16000.\n')
                        exit(1)
                    # wav 파일이 1채널(모노) 데이터 체크
                    if wav.getnchannels() != 1:
                        sys.stderr.write('This program \
                            supports monaural wav file only.\n')
                        exit(1)

                    num_samples = wav.getnframes()
                    waveform = wav.readframes(num_samples)
                    waveform = np.frombuffer(waveform, dtype=np.int16)

                    # FBANK를 계산(log_power: 로그 파워 정보도 출력되지만 여기서 사용하지 않기)
                    fbank, log_power = feat_extractor.ComputeFBANK(waveform)

                # 특징값 프레임 수와 차원 수 구하기
                (num_frames, num_dims) = np.shape(fbank)

                # 특징값 파일 이름(splitext로 확장자를 제거)
                out_file = os.path.splitext(os.path.basename(wav_path))[0]
                out_file = os.path.join(os.path.abspath(out_dir),
                                        out_file + '.bin')

                # 데이터를 float 32 형식으로 변환환
                fbank = fbank.astype(np.float32)

                # 데이터 파일로 출력
                fbank.tofile(out_file)
                # 발화 ID,  특징값 파일 경로, 프레임 수, 차원 수를 특징값 리스트에 쓰기
                file_feat.write("%s %s %d %d\n" %
                                (utterance_id, out_file, num_frames, num_dims))