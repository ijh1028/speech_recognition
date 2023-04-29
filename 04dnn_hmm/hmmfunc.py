import numpy as np
import json
import sys

class MonoPhoneHMM():
    ''' HMM 클래스
    MonophoneHMM정의
    Left-to-right
    공분산 행렬은 대각선 행렬을 가정
    '''

    def __init__(self):
        # 음소 리스트
        self.phones = []
        # 음소 수
        self.num_phones = 1
        # 각 음소 HMM의 상태수
        self.num_states = 1
        # GMM의 혼합수
        self.num_mixture = 1
        # 특징값 벡터의 차원수
        self.num_dims = 1
        # 정규 분포(Single Gaussian Model:SGM)의 파라미터
        self.pdf = None
        # 전이확률(대수치)
        self.trans = None
        # log(0)의 근사치
        self.LZERO = -1E10
        # 확률 계산에 가하는 값의 최소값
        # 계산량 삭감을 위해 이 값보다 작을 확률은 일부 계산에 있어서 무시당하다
        self.LSMALL = -0.5E10
        # 0의 근사치(값이 ZERO 이하라면, 로그는 LZERO로 대체한다)
        self.ZERO = 1E-100
        # 분산값의 플로어링 값
        self.MINVAR = 1E-4

        # 학습 시/인식 시 사용하는 파라미터
        # 정규 분포별로 계산되는 로그 확률
        self.elem_prob = None
        # 상태별로 계산되는 로그 확률
        self.state_prob = None
        # 전향 확률
        self.alpha = None
        # 역방향 확률
        self.beta = None
        # HMM우도
        self.loglikelihood = 0
        # 파라미터 갱신을 위한 변수
        self.pdf_accumulators = None
        self.trans_accumulators = None
        # 비타비 알고리즘 시 이용되는 누적 확률
        self.score = None
        # 비타비파스를 기억하는 행렬
        self.track = None
        # 비타비 알고리즘에 의한 점수
        self.viterbi_score = 0

    def make_proto(self,
                   phone_list,
                   num_states,
                   prob_loop,
                   num_dims):
        ''' HMM 프로토타입을 작성하다
        phone_list: 음소 목록
        num_states: 각 음소 HMM의 상태수
        prob_loop: 자기 루프 확률
        num_dims: 특징량의 차원수
        '''
        # 음소 리스트를 얻다
        self.phones = phone_list
        # 음소 수를 얻다
        self.num_phones = len(self.phones)
        # 각 음소 HMM의 상태 수를 얻다
        self.num_states = num_states
        # 특징량 벡터의 차원수 를 얻다
        self.num_dims = num_dims
        # GMM의 혼합 수는 1로 한다
        self.num_mixture = 1

        # 정규 분포 작성
        # 음소번호 p, 상태번호 s, 혼합요소번호 m의 정규 분포는 pdf[p][s][m]로 액세스한다
        # # pdf[p][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_s = []
                for m in range(self.num_mixture):
                    # 평균값 벡터 정의
                    mu = np.zeros(self.num_dims)
                    # 대각선 공분산 행렬의 대각선 성분 정의
                    var = np.ones(self.num_dims)
                    # 혼합수는 1이므로 혼합무게는 1.0
                    weight = 1.0
                    # gConst항 계산
                    gconst = self.calc_gconst(var)
                    # 정규 분포를 사전형으로 정의
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    # 정규 분포 추가
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            self.pdf.append(tmp_p)
        # 상태 전이 확률(의 로그치) 작성
        # 음소번호 p, 상태번호 s의 천이확률은
        # # trans[p][s] = [loop, next]
        # loop: 자기 루프 확률
        # next: 다음 상태로의 전이 확률

        # 다음 상태로 전이할 확률
        prob_next = 1.0 - prob_loop
        # 로그를 잡다
        log_prob_loop = np.log(prob_loop) \
            if prob_loop > self.ZERO else self.LZERO
        log_prob_next = np.log(prob_next) \
            if prob_next > self.ZERO else self.LZERO

        # self.trans에 저장
        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_trans = np.array([log_prob_loop,
                                      log_prob_next])
                tmp_p.append(tmp_trans)
            self.trans.append(tmp_p)

    def calc_gconst(self, variance):
        ''' gConst항(정규분포의 상수항)의 대수치)를 계산하다
        variance: 대각선 공분산 행렬의 대각선 성분
        '''
        gconst = self.num_dims * np.log(2.0 * np.pi) \
                 + np.sum(np.log(variance))
        return gconst

    def calc_pdf(self, pdf, obs):
        ''' 지정한 정규 분포에서의 로그 우도 계산
        pdf: 정규 분포
        obs: 입력 특징량
             한 프레임 분량의 벡터라도 프레임x차원배열로도입력가능
        logprob: 로그 우도
                1프레임분을 준 경우 스칼라값
                복수 프레임분을 준 경우는 프레임 몇 분의 크기를 가진 벡터
        '''
        # 정수항을 제외한 부분의 계산(exp(*)부분)
        tmp = (obs - pdf['mu']) ** 2 / pdf['var']
        if np.ndim(tmp) == 2:
            # obs가 [프레임x차원] 배열로 입력된 경우
            tmp = np.sum(tmp, 1)
        elif np.ndim(tmp) == 1:
            # obs가 1프레임 분량의 벡터로 입력된 경우
            tmp = np.sum(tmp)
        # 정수항을 붙여 -0.5를 곱하다
        logprob = -0.5 * (tmp + pdf['gConst'])
        return logprob

    def logadd(self, x, y):
        ''' x=log(a)와 y=log(b)에 대하여 log(a+b)를 계산하다
        x: log(a)
        y: log(b)
        z: log(a+b)
        '''
        if x > y:
            z = x + np.log(1.0 + np.exp(y - x))
        else:
            z = y + np.log(1.0 + np.exp(x - y))
        return z

    def flat_init(self, mean, var):
        ''' 플랫 스타트에 의한 초기화
        학습 데이터 전체의 평균 분산을 HMM의 전체 정규 분포의 파라미터로 하다
        mean: 학습 데이터 전체의 평균 벡터
        var: 학습 데이터 전체의 대각선 공분산
        '''
        # 차원 수가 일치하지 않는 경우 오류
        if self.num_dims != len(mean) or \
                self.num_dims != len(var):
            sys.stderr.write('flat_init: invalid mean or var\n')
            return 1
        for p in range(self.num_phones):
            for s in range(self.num_states):
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    pdf['mu'] = mean
                    pdf['var'] = var
                    pdf['gConst'] = self.calc_gconst(var)

    def calc_out_prob(self, feat, label):
        ''' 출력 확률 계산
        feat: 1발화분 특징량[프레임수 x 차원수]
        label 1발화분의 라벨
        '''
        # 특징값의 프레임 수를 얻다
        feat_len = np.shape(feat)[0]
        # 라벨의 길이를 얻다
        label_len = len(label)

        # 정규 분포별로 계산되는 로그 확률
        self.elem_prob = np.zeros((label_len,
                                   self.num_states,
                                   self.num_mixture,
                                   feat_len))

        # 각 상태 (q, s)에서 시각 t의 출력 확률
        # (state_prob = sum(weight*elem_prob))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        # elem_prob, state_prob 를 계산
        # l: 라벨상에서 몇 번째 음소인가
        # p: l이 음소 목록상에서 어떤 음소인가
        # s: 상태
        # t: 프레임
        # m: 혼합 요소
        for l, p in enumerate(label):
            for s in range(self.num_states):
                # state_prob를log(0)로 초기화
                self.state_prob[l][s][:] = \
                    self.LZERO * np.ones(feat_len)
                for m in range(self.num_mixture):
                    # 정규 분포를 추출
                    pdf = self.pdf[p][s][m]
                    # 확률 계산
                    self.elem_prob[l][s][m][:] = \
                        self.calc_pdf(pdf, feat)
                    # GMM의 가중치를 더하기
                    tmp_prob = np.log(pdf['weight']) \
                               + self.elem_prob[l][s][m][:]
                    # 확률을 더하기
                    for t in range(feat_len):
                        self.state_prob[l][s][t] = \
                            self.logadd(self.state_prob[l][s][t],
                                        tmp_prob[t])

    def calc_alpha(self, label):
        ''' 전향 확률 알파를 구하다
        left-to-right형 HMM을 전제로 한 실장에 되어 있다
        label: 라벨
        '''
        # 라벨 길이와 프레임 수를 얻다
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # alpha를 log(0)로 초기화
        self.alpha = self.LZERO * np.ones((label_len,
                                           self.num_states,
                                           feat_len))

        # t=0일 때,
        # 반드시 첫 음소의 첫 번째 상태에 있다
        self.alpha[0][0][0] = self.state_prob[0][0][0]

        # t: 프레임
        # l: 라벨 위의 몇 번째 음소인가
        # p: l이 음소 리스트 상의 어느 음소인지
        # s: 상태
        for t in range(1, feat_len):
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    # 자기 루프의 고려
                    self.alpha[l][s][t] = \
                        self.alpha[l][s][t - 1] \
                        + self.trans[p][s][0]
                    if s > 0:
                        # 선두 상태가 아니라면 하나 전 상태로부터의 전이를 고려
                        tmp = self.alpha[l][s - 1][t - 1] \
                              + self.trans[p][s - 1][1]
                        # self 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            self.alpha[l][s][t] = \
                                self.logadd(self.alpha[l][s][t],
                                            tmp)
                    elif l > 0:
                        # 선두 음소가 아니라 선두 상태인 경우
                        # 하나 전 음소의 종단 상태에서 전이
                        prev_p = label[l - 1]
                        tmp = self.alpha[l - 1][-1][t - 1] \
                              + self.trans[prev_p][-1][1]
                        # self 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            self.alpha[l][s][t] = \
                                self.logadd(self.alpha[l][s][t],
                                            tmp)
                    # # else:
                    #   # 선두 음소이면서 선두 상태인 경우
                    #   # self 루프 이외의 전이는 없다

                    # state_prob 추가
                    self.alpha[l][s][t] += \
                        self.state_prob[l][s][t]

        # HMM의 로그 우도는 종단 alpha
        self.loglikelihood = self.alpha[-1][-1][-1]

    def calc_beta(self, label):
        ''' 역방향 확률 beta를 구하다
            left-to-right형 HMM을 전제로 한 실장에 되어 있다
        label: 라벨
        '''
        # 라벨 길이와 프레임 수를 얻다
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # alpha를 log(0)로 초기화
        self.beta = self.LZERO * np.ones((label_len,
                                          self.num_states,
                                          feat_len))

        # t=-1(최종 프레임)일 때, 반드시 마지막 음소의 마지막 상태에 있다
        # (확률은 log(1) = 0)
        self.beta[-1][-1][-1] = 0.0

        # t: 프레임
        # l: 라벨 위의 몇 번째 음소인가
        # p: l이 음소 리스트 상의 어느 음소인지
        # s: 상태
        # calc_alpha와 달리 t는 feat_len-2에서 0으로 나아가는 점에 주의
        for t in range(0, feat_len - 1)[::-1]:
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    # self 루프의 고려
                    self.beta[l][s][t] = \
                        self.beta[l][s][t + 1] \
                        + self.trans[p][s][0] \
                        + self.state_prob[l][s][t + 1]
                    if s < self.num_states - 1:
                        # 종단 상태가 아니라면 하나 후 상태로의 전이를 고려
                        tmp = self.beta[l][s + 1][t + 1] \
                              + self.trans[p][s][1] \
                              + self.state_prob[l][s + 1][t + 1]
                        # self 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            self.beta[l][s][t] = \
                                self.logadd(self.beta[l][s][t],
                                            tmp)
                    elif l < label_len - 1:
                        # 종단의 음소가 아니라 또한 종단 상태인 경우
                        # 하나 후 음소의 선두 상태로의 전환
                        tmp = self.beta[l + 1][0][t + 1] \
                              + self.trans[p][s][1] \
                              + self.state_prob[l + 1][0][t + 1]
                        # self 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            self.beta[l][s][t] = \
                                self.logadd(self.beta[l][s][t],
                                            tmp)
                    # # else:
                    #   # 종단 음소이면서 종단 상태인 경우
                    #   # 자기 루프 이외의 전이는 없다

    def reset_accumulators(self):
        ''' accumulators (파라미터 갱신에 필요한 변수)을 초기화하다
        '''
        # GMM을 업데이트하기 위한 accumulators
        self.pdf_accumulators = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_s = []
                for m in range(self.num_mixture):
                    pdf_stats = {}
                    pdf_stats['weight'] = \
                        {'num': self.LZERO,
                         'den': self.LZERO}
                    pdf_stats['mu'] = \
                        {'num': np.zeros(self.num_dims),
                         'den': self.LZERO}
                    pdf_stats['var'] = \
                        {'num': np.zeros(self.num_dims),
                         'den': self.LZERO}
                    tmp_s.append(pdf_stats)
                tmp_p.append(tmp_s)
            self.pdf_accumulators.append(tmp_p)

        # 전이 확률을 업데이트하기 위한 accumulators
        self.trans_accumulators = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                trans_stats = \
                    {'num': np.ones(2) * self.LZERO,
                     'den': self.LZERO}
                tmp_p.append(trans_stats)
            self.trans_accumulators.append(tmp_p)

    def update_accumulators(self, feat, label):
        ''' accumulators 업데이트
        left-to-right를 전제로 한 실장으로 되어 있다.
        feat: 특징값
        label: 라벨
        '''
        # 라벨 길이 가져오기
        label_len = len(label)
        # 프레임 수 취득
        feat_len = np.shape(feat)[0]

        # t: 프레임
        # l: 라벨 위의 몇 번째 음소인가
        # p: l이 음소 리스트 상의 어느 음소인지
        # s: 상태
        for t in range(feat_len):
            for l in range(label_len):
                p = label[l]
                for s in range(self.num_states):
                    if t == 0 and l == 0 and s == 0:
                        # t=0일때는 반드시 선두상태
                        # (대수확률이므로 log(1)=0)
                        lconst = 0
                    elif t == 0:
                        # t=0으로 선두상태가 아니라면
                        # 확률이 제로이므로 건너뛰기
                        continue
                    elif s > 0:
                        # t>0에서 선두 상태가 아닌 경우
                        # self 루프
                        lconst = self.alpha[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 하나 전 상태로부터의 전이를 고려
                        tmp = self.alpha[l][s - 1][t - 1] \
                              + self.trans[p][s - 1][1]
                        # self 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            lconst = self.logadd(lconst,
                                                 tmp)
                    elif l > 0:
                        # t>0 선두 음소가 아니라, 선두 상태인 경우
                        # self 루프
                        lconst = self.alpha[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 하나 전 음소의 종단 상태에서 전이
                        prev_p = label[l - 1]
                        tmp = self.alpha[l - 1][-1][t - 1] \
                              + self.trans[prev_p][-1][1]
                        # self 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            lconst = self.logadd(lconst,
                                                 tmp)
                    else:
                        # 선두 음소이면서 선두 상태인 경우
                        lconst = self.alpha[l][s][t - 1] \
                                 + self.trans[p][s][0]

                    # 후진 확률과 1/P를 더하다
                    lconst += self.beta[l][s][t] \
                              - self.loglikelihood
                    # accumulators 업데이트
                    for m in range(self.num_mixture):
                        pdf = self.pdf[p][s][m]
                        L = lconst \
                            + np.log(pdf['weight']) \
                            + self.elem_prob[l][s][m][t]

                        pdf_accum = self.pdf_accumulators[p][s][m]
                        # 평균값 벡터 갱신식 분자는 로그를 차지하지 않다
                        pdf_accum['mu']['num'] += \
                            np.exp(L) * feat[t]
                        # 분모는 대수상으로 갱신
                        if L > self.LSMALL:
                            pdf_accum['mu']['den'] = \
                                self.logadd(pdf_accum['mu']['den'],
                                            L)
                        # 대각선 공분산 갱신식 분자는 로그를 차지하지 않다
                        dev = feat[t] - pdf['mu']
                        pdf_accum['var']['num'] += \
                            np.exp(L) * (dev ** 2)
                        # 분모는 평균값의 것과 같은 값
                        pdf_accum['var']['den'] = \
                            pdf_accum['mu']['den']

                        # GMM 가중치 갱신식 분자는 평균 분산의 분모와 같은 값
                        pdf_accum['weight']['num'] = \
                            pdf_accum['mu']['den']

        # 전이 확률의 accumulators와 GMM 가중치 accumulators 분모 업데이트
        for t in range(feat_len):
            for l in range(label_len):
                p = label[l]
                for s in range(self.num_states):
                    # GMM 가중치 accumulator의 분모와 전이 확률 accumulator의 분모 갱신에 이용하다
                    alphabeta = self.alpha[l][s][t] \
                                + self.beta[l][s][t] \
                                - self.loglikelihood

                    # GMM 가중치 accumulator 분모 업데이트
                    for m in range(self.num_mixture):
                        pdf_accum = \
                            self.pdf_accumulators[p][s][m]
                        # 분모는 모든 m에서 같은 값이므로
                        # m==0일때만 계산
                        if m == 0:
                            if alphabeta > self.LSMALL:
                                pdf_accum['weight']['den'] = \
                                    self.logadd( \
                                        pdf_accum['weight']['den'],
                                        alphabeta)
                        else:
                            tmp = self.pdf_accumulators[p][s][0]
                            pdf_accum['weight']['den'] = \
                                tmp['weight']['den']

                    # 전이 확률 accumulator 분모 업데이트
                    trans_accum = self.trans_accumulators[p][s]
                    if t < feat_len - 1 \
                            and alphabeta > self.LSMALL:
                        trans_accum['den'] = \
                            self.logadd(trans_accum['den'],
                                        alphabeta)


                    # 전이 확률 accumulator의 분자 갱신
                    if t == feat_len - 1:
                        # 최종 프레임은 건너뛰기
                        continue
                    elif s < self.num_states - 1:
                        # 각 음소의 비종단 상태인 경우
                        # self 루프
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][0] \
                              + self.state_prob[l][s][t + 1] \
                              + self.beta[l][s][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)

                        # 천이
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][1] \
                              + self.state_prob[l][s + 1][t + 1] \
                              + self.beta[l][s + 1][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][1] = \
                                self.logadd(trans_accum['num'][1],
                                            tmp)
                    elif l < label_len - 1:
                        # 종단 상태이면서 비종단 음소
                        # self 루프
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][0] \
                              + self.state_prob[l][s][t + 1] \
                              + self.beta[l][s][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)
                        # 다음 음소의 시작 상태로의 전환
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][1] \
                              + self.state_prob[l + 1][0][t + 1] \
                              + self.beta[l + 1][0][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][1] = \
                                self.logadd(trans_accum['num'][1],
                                            tmp)
                    else:
                        # 최종 상태
                        # self 루프
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][0] \
                              + self.state_prob[l][s][t + 1] \
                              + self.beta[l][s][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)

    def update_parameters(self):
        ''' 파라미터 갱신
        '''
        for p in range(self.num_phones):
            for s in range(self.num_states):
                # 전이 확률 업데이트
                trans_accum = self.trans_accumulators[p][s]
                self.trans[p][s] = \
                    trans_accum['num'] - trans_accum['den']
                # 확률 총합이 1이 되도록 정규화
                tmp = self.logadd(self.trans[p][s][0],
                                  self.trans[p][s][1])
                self.trans[p][s] -= tmp
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    pdf_accum = self.pdf_accumulators[p][s][m]
                    # 평균값 벡터 업데이트
                    den = np.exp(pdf_accum['mu']['den'])
                    if den > 0:
                        pdf['mu'] = pdf_accum['mu']['num'] / den
                    # 대각선 공분산 갱신
                    den = np.exp(pdf_accum['var']['den'])
                    if den > 0:
                        pdf['var'] = pdf_accum['var']['num'] / den
                    # 분산 플로어링
                    pdf['var'][pdf['var'] < self.MINVAR] = \
                        self.MINVAR
                    # gConst항의 갱신
                    gconst = self.calc_gconst(pdf['var'])
                    pdf['gConst'] = gconst

                    # GMM 가중치 업데이트
                    tmp = pdf_accum['weight']['num'] - \
                          pdf_accum['weight']['den']
                    pdf['weight'] = np.exp(tmp)
                # GMM 가중치의 총합이 1이 되도록 정규화
                wsum = 0.0
                for m in range(self.num_mixture):
                    wsum += self.pdf[p][s][m]['weight']
                for m in range(self.num_mixture):
                    self.pdf[p][s][m]['weight'] /= wsum

    def viterbi_decoding(self, label):
        ''' 비타비 알고리즘에 의한 디코딩
            left-to-right형 HMM을 전제로 한 실장에 되어 있다
        lable: 라벨
        '''
        # 라벨 길이와 프레임 수를 얻다
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # score를 log(0)로 초기화
        self.score = self.LZERO * np.ones((label_len,
                                           self.num_states,
                                           feat_len))
        # 백트랙용 전이 기록 영역
        # 0 : 자기 루프 1 : 다음 상태로 전이
        self.track = np.zeros((label_len,
                               self.num_states,
                               feat_len), np.int16)
        # t=0일 때, 반드시 첫 음소의 첫 번째 상태에 있다
        self.score[0][0][0] = self.state_prob[0][0][0]

        # t: 프레임
        # l: 라벨 위의 몇 번째 음소인가
        # p: l이 음소 리스트 상의 어느 음소인지
        # s: 상태
        for t in range(1, feat_len):
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    if s > 0:
                        # 선두 상태가 아니라면 한 가지 전 상태로부터의 전이인가
                        # self 루프 중 하나
                        p_next = self.score[l][s - 1][t - 1] \
                                 + self.trans[p][s - 1][1]
                        p_loop = self.score[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 큰 쪽을 채용
                        cand = [p_loop, p_next]
                        tran = np.argmax(cand)
                        self.score[l][s][t] = cand[tran]
                        self.track[l][s][t] = tran
                    elif l > 0:
                        # 선두 음소가 아니라선두 상태인 경우
                        # 하나 전 음소의 종단 상태에서 전이인가
                        # self 루프 중 하나
                        prev_p = label[l - 1]
                        p_next = self.score[l - 1][-1][t - 1] \
                                 + self.trans[prev_p][-1][1]
                        p_loop = self.score[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 큰 쪽을 채용
                        cand = [p_loop, p_next]
                        tran = np.argmax(cand)
                        self.score[l][s][t] = cand[tran]
                        self.track[l][s][t] = tran
                    else:
                        # 선두 음소이면서 선두 상태인 경우
                        # self 루프만
                        p_loop = self.score[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        self.score[l][s][t] = p_loop
                        self.track[l][s][t] = 0

                    # state_prob 추가
                    self.score[l][s][t] += \
                        self.state_prob[l][s][t]

        # 비타비 스코어 종단의 score
        self.viterbi_score = self.score[-1][-1][-1]

    def back_track(self):
        ''' 비타비패스 백트랙
        viterbi_path: 백트랙의 결과
        '''
        # 라벨 길이와 프레임 수를 얻다
        (label_len, _, feat_len) = np.shape(self.track)

        viterbi_path = []
        # 종단부터 시작
        l = label_len - 1  # 음소
        s = self.num_states - 1  # 상태
        t = feat_len - 1  # 프레임
        while True:
            viterbi_path.append([l, s, t])
            # 시작점에 도달하면 종료
            if l == 0 and s == 0 and t == 0:
                break
            # track 값을 보다
            # 0이면 self 루프, 1이면 전이
            tran = self.track[l][s][t]

            if tran == 1:
                # 천이
                if s == 0:
                    # 이전 음소로부터의 전이
                    # l을 줄여서 s를 종단
                    l = l - 1
                    s = self.num_states - 1
                else:
                    # 같은 음소의 이전 상태로부터의 전이
                    # s를 줄이다
                    s = s - 1
            # t를 줄이다
            t = t - 1

        # viterbi_path를 역순으로 정렬
        viterbi_path = viterbi_path[::-1]
        return viterbi_path

    def mixup(self):
        ''' HMM의 혼합수를 두 배로 늘리다
        '''

        for p in range(self.num_phones):
            for s in range(self.num_states):
                pdf = self.pdf[p][s]
                for m in range(self.num_mixture):
                    # 혼합 가중치를 얻다
                    weight = pdf[m]['weight']
                    # 혼합수를 두 배로 늘린 만큼 무게를 0.5배로 늘리다
                    weight *= 0.5
                    # 복사원의 혼합 가중치도 0.5배 한다
                    pdf[m]['weight'] *= 0.5
                    # gConst항을 얻다
                    gconst = pdf[m]['gConst']

                    # 평균값 벡터를 얻다
                    mu = pdf[m]['mu'].copy()
                    # 대각 공분산을 얻다
                    var = pdf[m]['var'].copy()

                    # 표준 편차를 얻다
                    std = np.sqrt(var)
                    # 표준 편차의 0.2배를 평균값 벡터에 더하다
                    mu = mu + 0.2 * std
                    # 복사원의 평균값 벡터는 0.2*std로 빼기
                    pdf[m]['mu'] = pdf[m]['mu'] - 0.2 * std

                    # 정규 분포를 사전형으로 정의
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    # 정규 분포를 가하다
                    pdf.append(gaussian)

        # GMM의 혼합수를 두 배로 늘리다
        self.num_mixture *= 2

    def train(self, feat_list, label_list, report_interval=10):
        ''' HMM을 1iteration 분 갱신
        feat_list:  특징값 파일 목록
                    발화 ID를 key, 특징량 파일 경로를 value로 하는 사전
        label_list: 라벨 리스트
                    발화 ID를 key, 라벨을 value로 한다.
                    사전
        report_interval: 처리 도중 결과를 표시하는 간격(발화수)
        '''
        # accumulators(파라미터 갱신에 사용하는 변수)를 리셋
        self.reset_accumulators()

        # 특징값 파일을 하나씩 열어서 처리
        count = 0
        ll_per_utt = 0.0
        partial_ll = 0.0
        for utt, ff in feat_list.items():
            # 처리한 발화수를 1 늘리다
            count += 1
            # 특징값 파일 열기
            feat = np.fromfile(ff, dtype=np.float32)
            # 프레임 수 x 차원 수 배열로 변형
            feat = feat.reshape(-1, self.num_dims)
            # 라벨 취득
            label = label_list[utt]

            # 각 분포의 출력 확률을 구하다
            self.calc_out_prob(feat, label)
            # 전향 확률을 찾다
            self.calc_alpha(label)
            # 뒤돌아설 확률을 구하다
            self.calc_beta(label)
            # accumulators를 갱신
            self.update_accumulators(feat, label)
            # 로그 우도 가산
            ll_per_utt += self.loglikelihood

            # 도중 결과를 표시
            partial_ll += self.loglikelihood
            if count % report_interval == 0:
                partial_ll /= report_interval
                print('  %d / %d utterances processed' \
                      % (count, len(feat_list)))
                print('  log likelihood averaged' \
                      ' over %d utterances: %f' \
                      % (report_interval, partial_ll))

        # 모델 파라미터 갱신
        self.update_parameters()
        # 로그 우도의 발화 평균을 구하다
        ll_per_utt /= count
        print('average log likelihood: %f' % (ll_per_utt))

    def recognize(self, feat, lexicon):
        ''' 고립 단어 인식
        feat:    특징값
        lexicon: 인식 단어 목록. 아래의 사서형이 리스트에 나와 있다
                 {'word':단어,
                  'pron':음소열,
                  'int':음소열 수치 표기}
        '''
        # 단어 목록 내의 단어마다 우도를 계산
        # 결과 목록
        result = []
        for lex in lexicon:
            # 음소열의 수치 표기를 얻다
            label = lex['int']
            # 각 분포의 출력 확률을 구하다
            self.calc_out_prob(feat, label)
            # 비타비 알고리즘 실행
            self.viterbi_decoding(label)
            result.append({'word': lex['word'],
                           'score': self.viterbi_score})

        # 스코어의 승순으로 정렬
        result = sorted(result,
                        key=lambda x: x['score'],
                        reverse=True)
        # 인식 결과와 점수 정보를 반환
        return (result[0]['word'], result)

    def set_out_prob(self, prob, label):
        ''' 출력 확률을 설정
        prob DNN이 출력한 확률은 [프레임 수 * (음소 수 *상태 수)]의 2차원 배열로 저장
        label 1발화만큼의 라벨
        '''
        # 프레임 개수를 구한다
        feat_len = np.shape(prob)[0]
        # 라벨 길이를 구한다
        label_len = len(label)

        # 각 상태 (q, s)에서 시각 t의 출력 확률
        # (state_prob = sum(weight*elem_prob))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        # state_prob를 계산
        # l: 라벨상에서 몇 번째 음소인가
        # p: l이 음소 목록상에서 어떤 음소인가
        # s: 상태
        # t: 프레임
        for l, p in enumerate(label):
            for s in range(self.num_states):
                # 음소 p의 상태 s 값은 DNN 출력상에서
                # p*num_states+s 에 저장되어 있다
                state = p * self.num_states + s
                for t in range(feat_len):
                    self.state_prob[l][s][t] = \
                        prob[t][state]

    def recognize_with_dnn(self, prob, lexicon):
        ''' DNN이 출력한 확률을 이용하여 고립단어 인식을 수행
        prob:    DNN의 출력 확률
                 (단, 각 상태의 사전 확률로 나눈 빈도로 변환해둘 것)
        lexicon: 인식 단어 리스트
                 아래와 같이 사전형 목록으로 되어 있다
                 {'word':단어,
                  'pron':음소열,
                  'int':음소열 수치 표기}
        '''
        # 단어 리스트의 단어별 빈도를 계산한 결과 리스트
        result = []
        for lex in lexicon:
            # 음소열의 수치값 표기를 구한다
            label = lex['int']
            # 각 분포의 출력 확률을 설정
            self.set_out_prob(prob, label)
            # 비터비 알고리즘 실행
            self.viterbi_decoding(label)
            result.append({'word': lex['word'],
                           'score': self.viterbi_score})

        # 점수가 높은 순으로 나열
        result = sorted(result,
                        key=lambda x: x['score'],
                        reverse=True)
        # 인식 결과와 점수 정보를 반환
        return (result[0]['word'], result)

    def phone_alignment(self, feat, label):
        ''' 음소 얼라인먼트를 실시
        feat: 특징값
        label: 라벨
        '''
        # 각 분포의 출력 확률을 구하다
        self.calc_out_prob(feat, label)
        # 비타비 알고리즘 실행
        self.viterbi_decoding(label)
        # 백트랙 실행
        viterbi_path = self.back_track()
        # 비타비 패스에서 프레임별 음소열로 변환
        phone_alignment = []
        for vp in viterbi_path:
            # 라벨 상의 음소 인덱스 가져오기
            l = vp[0]
            # 음소 번호를 음소 목록 상의 번호로 변환
            p = label[l]
            # 번호에서 음소 기호로 변환
            ph = self.phones[p]
            # phone_ alignment 말미에 추가
            phone_alignment.append(ph)

        return phone_alignment

    def state_alignment(self, feat, label):
        ''' HMM 상태에서의 얼라인먼트를 실시
        feat: 특징값
        label: 라벨
        state_alignment: 프레임별 상태 번호
            단, 여기서의 상태 번호는 (음소번호)*(상태수)+(음소내상태번호) 라고 하자
        '''
        # 각 분포의 출력 확률을 구하다
        self.calc_out_prob(feat, label)
        # 비타비 알고리즘 실행
        self.viterbi_decoding(label)
        # 백트랙 실행
        viterbi_path = self.back_track()
        # 비타비 패스에서 프레임별 상태 번호열로 변환
        state_alignment = []
        for vp in viterbi_path:
            # 라벨 상의 음소 인덱스 가져오기
            l = vp[0]
            # 음소 번호를 음소 목록 상의 번호로 변환
            p = label[l]
            # 음소 내 상태 번호 취득
            s = vp[1]
            # 출력 시 상태 번호는 p * num_states + s 로 하자
            state = p * self.num_states + s
            # phone_ alignment 마지막에 추가
            state_alignment.append(state)

        return state_alignment

    def save_hmm(self, filename):
        ''' HMM 파라미터를 json 형식으로 저장
        filename: 저장 파일명
        '''
        # json 형식으로 저장하기 위해 HMM의 정보를 사전 형식으로 변환하다
        hmmjson = {}
        # 기본 정보 입력
        hmmjson['num_phones'] = self.num_phones
        hmmjson['num_states'] = self.num_states
        hmmjson['num_mixture'] = self.num_mixture
        hmmjson['num_dims'] = self.num_dims
        # 음소 모형 목록
        hmmjson['hmms'] = []
        for p, phone in enumerate(self.phones):
            model_p = {}
            # 음소명
            model_p['phone'] = phone
            # HMM 리스트
            model_p['hmm'] = []
            for s in range(self.num_states):
                model_s = {}
                # 상태 번호
                model_s['state'] = s
                # 전이 확률(대수치에서 되돌림)
                model_s['trans'] = \
                    list(np.exp(self.trans[p][s]))
                # GMM 리스트
                model_s['gmm'] = []
                for m in range(self.num_mixture):
                    model_m = {}
                    # 혼합 요소 번호
                    model_m['mixture'] = m
                    # 혼합 가중치
                    model_m['weight'] = \
                        self.pdf[p][s][m]['weight']
                    # 평균값 벡터
                    # json은 ndarray를 다룰 수 없기 때문에 list형으로 변환해 두다
                    model_m['mean'] = \
                        list(self.pdf[p][s][m]['mu'])
                    # 대각선 공분산
                    model_m['variance'] = \
                        list(self.pdf[p][s][m]['var'])
                    # gConst
                    model_m['gConst'] = \
                        self.pdf[p][s][m]['gConst']
                    # gmm 리스트에 추가
                    model_s['gmm'].append(model_m)
                # hmm 리스트에 추가
                model_p['hmm'].append(model_s)
            # 음소 모델 리스트에 추가
            hmmjson['hmms'].append(model_p)

        # JSON 형식으로 저장
        with open(filename, mode='w') as f:
            json.dump(hmmjson, f, indent=4)

    def load_hmm(self, filename):
        ''' json 형식의 HMM 파일 가져오기
        filename: 가져오기 파일명
        '''
        # JSON 형식의 HMM 파일 가져오기
        with open(filename, mode='r') as f:
            hmmjson = json.load(f)

        # 사전 값을 읽어 나가다
        self.num_phones = hmmjson['num_phones']
        self.num_states = hmmjson['num_states']
        self.num_mixture = hmmjson['num_mixture']
        self.num_dims = hmmjson['num_dims']

        # 음소 정보 가져오기
        self.phones = []
        for p in range(self.num_phones):
            hmms = hmmjson['hmms'][p]
            self.phones.append(hmms['phone'])

        # 전이 확률 읽기
        # 음소번호 p, 상태번호 s의 천이확률은
        # trans[p][s] = [loop, next]
        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                hmm = hmms['hmm'][s]
                # 전이 확률 읽기
                tmp_trans = np.array(hmm['trans'])
                # 총합이 1이 되도록 정규화
                tmp_trans /= np.sum(tmp_trans)
                # 로그로 변환
                for i in [0, 1]:
                    tmp_trans[i] = np.log(tmp_trans[i]) \
                        if tmp_trans[i] > self.ZERO \
                        else self.LZERO
                tmp_p.append(tmp_trans)
            # self.trans에 추가
            self.trans.append(tmp_p)

        # 정규 분포 파라미터 가져오기
        # 음소번호 p, 상태번호 s, 혼합요소번호 m
        # 의 정규 분포는 pdf[p][s][m]로 액세스한다
        # # pdf[p][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                tmp_s = []
                hmm = hmms['hmm'][s]
                for m in range(self.num_mixture):
                    gmm = hmm['gmm'][m]
                    # 무게, 평균, 분산, gConst를 취득
                    weight = gmm['weight']
                    mu = np.array(gmm['mean'])
                    var = np.array(gmm['variance'])
                    gconst = gmm['gConst']
                    # 정규 분포 작성
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            # self.pdf에 추가
            self.pdf.append(tmp_p)