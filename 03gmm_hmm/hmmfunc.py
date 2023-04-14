import numpy as np
import json
import sys

# MonoPhoneHMM Class 생성자
class MonoPhoneHMM():
    '''
    HMM Class MonoPhoneHMM을 정의
    Left-to-right 형
    공분산 행렬은 대각 행렬로 가정
    '''

    def __init__(self):
        # 음소 목록
        self.phones = []
        # 음소 수
        self.num_phones = 1
        # 각 음소 HMM의 상태 수
        self.num_states = 1
        # GMM 혼합 수
        self.num_mixture = 1
        # 특징값 벡터 차원 수
        self.num_dims = 1
        # 정규분포 (Single Gaussian Model: SGM)의 매개변수
        self.pdf = None
        # 전이 확률(로그값)
        self.trans = None
        # log(0) 근사값
        self.LZERO = -1E10
        # 확률 계산에 더하는 값의 최솟값
        # 효율적인 계산을 위해, 이 값보다 작은 확률은 일부 계산에서 무시
        self.LSMALL = -0.5E10
        # 0 근사값(값이 zero 이하면 로그는 LZERO로 치환)
        self.ZERO = 1E-100
        # 분산값 Flooring 값
        self.MINVAR = 1E-4

        # 학습 및 인식 시에 사용하는 매개변수
        # 정규분포별로 계산되는 로그 확률
        self.elem_prob = None
        # 상태별로 계산되는 로그 확률
        self.state_prob = None
        # 전향 확률
        self.alpha = None
        # 후향 확률
        self.beta = None
        # HMM 빈도
        self.loglikelihood = 0
        # 매개변수 갱신을 위한 변수
        self.pdf_accumlators = None
        self.trans_accumulators = None
        # 비터비 알고리즘에 사용하는 누적 확률
        self.score = None
        # 비터비 경로를 저장하는 행렬
        self.track = None
        # 비터비 알고리즘에 의한 스코어
        self.viterbi_score = 0


# MonoPhoneHMM 클래스 프로토타입 작성 및 HMM 저장
    def make_proto(self,
                   phone_list, num_states,
                   prob_loop, num_dims):
        '''
        HMM 프로토타입을 생성한다

        phone_list : 음소 목록
        num_states : 각 음소의 HMM 상태 수
        prob_loop : 자기 루프 확률
        num_dims : 특징값의 차원 수
        '''
        # 음소 목록을 기록
        self.phones = phone_list
        # 음소 수를 기록
        self.num_phones = len(self.phones)
        # 각 음소의 HMM 상태 수를 기록
        self.num_states = num_states
        # 특징값 벡터 차원 수를 기록
        self.num_dims = num_dims
        # GMM 상태 수는 1로 한다
        self.num_mixture = 1

        # 정규 분포 생성
        # 음소 번호 p, 상태 번호 s, 혼합 요소 번호 m의 정규분포는 pdf[p][s][m]이다
        # pdf[p][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_s = []
                for m in range(self.num_mixture):
                    # 평균값 벡터 정의
                    mu = np.zeros(self.num_dims)
                    # 대각 공분산 행렬의 대각 성분 정의
                    var = np.ones(self.num_dims)
                    # 혼합 수는 1이므로 혼합 가중치는 1.0
                    weight = 1.0
                    # gConst 항을 계산
                    gconst = self.calc_gconst(var)
                    # 정규분포를 사전형으로 정의
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    # 정규분포를 추가
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            self.pdf.append(tmp_p)

        # 상태 천이 확률(로그값)을 생성
        # 음소 번호 p, 상태 번호 s의 천이 확률은
        # trans[p][s] = [loop, next]
        # loop : 자기 루프 확률
        # next : 다음 상태의 천이 확률

        # 다음 상태로 천이할 확률
        prob_next = 1.0 - prob_loop
        # 로그를 취한다
        log_prob_loop = np.log(prob_loop) if prob_loop > self.ZERO else self.LZERO
        log_prob_next = np.log(prob_next) if prob_next > self.ZERO else self.LZERO

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
        '''
        gConst 항(정규분포 정수항 로그값)을 계산
        variance : 대각 공분산 행렬 대각 성분
        '''
        gconst = self.num_dims * np.log(2.0 * np.pi) + np.sum(np.log(variance))
        return gconst
    
    # MonoPhoneHMM 클래스의 HMM 학습 부분
    def calc_pdf(self, pdf, obs):
        ''' 지정한 정규분포에서 로그 빈도 계산
        pdf:     정규분포
        obs:     입력 특징값
                 1프레임 분량의 벡터도
                 프레임 x 차원 배열도 입력 가능
        logprob: 로그빈도
                 1프레임 분량이 제공된 경우에는 스칼라 값
                 복수 프레임 분량이 제공된 경우에는 프레임 수만큼의 크기를 가진 벡터
        '''
        # 정수항을 제외한 부분 계산(exp(*)의 부분)
        tmp = (obs - pdf['mu']) ** 2 / pdf['var']
        if np.ndim(tmp) == 2:
            # obs가 [프레임 x 차원] 배열로 입력된 경우
            tmp = np.sum(tmp, 1)
        elif np.ndim(tmp) == 1:
            # obs가ㅏ 1프레임 분량 벡터로 입력된 경우
            tmp = np.sum(tmp)
        # 정수항을 더해서 -0.5를 곱한다
        logprob = -0.5 * (tmp + pdf['gConst'])
        return logprob

    def logadd(self, x, y):
        ''' x=log(a)와 y=log(b)에 대해
            log(a+b)를 계산한다
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
        # 차원 수가 일치하지 않는 경우에는 에러
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
        feat: 1발화 분량의 특징값 [프레임 수 x 차원 수]
        label 1발화 분량 라벨
        '''
        # 특징값 프레임 수를 기록
        feat_len = np.shape(feat)[0]
        # 라벨 길이를 기록
        label_len = len(label)

        # 정규분포별로 계산되는 로그 확률
        self.elem_prob = np.zeros((label_len,
                                   self.num_states,
                                   self.num_mixture,
                                   feat_len))

        # 각 상태(q,s)에서 시각 t 출력 확률
        # (state_prob = sum(weight*elem_prob))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        # elem_prob, state_prob를 계산
        # l: 라벨에서 몇 번째 음소인가
        # p: l(라벨)이 음소 목록상 어느 음소에 해당하는가
        # s: 상태
        # t: 프레임
        # m: 혼합요소
        for l, p in enumerate(label):
            for s in range(self.num_states):
                # state_prob를 log(0)으로 초기화
                self.state_prob[l][s][:] = \
                    self.LZERO * np.ones(feat_len)
                for m in range(self.num_mixture):
                    # 정규분포를 추출
                    pdf = self.pdf[p][s][m]
                    # 확률 계산
                    self.elem_prob[l][s][m][:] = \
                        self.calc_pdf(pdf, feat)
                    # GMM 가중치를 더한다
                    tmp_prob = np.log(pdf['weight']) \
                               + self.elem_prob[l][s][m][:]
                    # 확률을 더한다
                    for t in range(feat_len):
                        self.state_prob[l][s][t] = \
                            self.logadd(self.state_prob[l][s][t],
                                        tmp_prob[t])

    def calc_alpha(self, label):
        ''' 전향 확률 alpha를 구한다
            left-to-right형 HMM을 전제로 구현한다
        label: 라벨
        '''
        # 라벨 길이와 프레임 수를 기록
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # alpha를 log(0) 으로 초기화
        self.alpha = self.LZERO * np.ones((label_len,
                                           self.num_states,
                                           feat_len))

        # t = 0일 경우
        # 반드시 최초 음소의 최초 상태에 위차
        self.alpha[0][0][0] = self.state_prob[0][0][0]

        # t: 프레임
        # l: 라벨에서 몇 번째 음소인가
        # p: l(라벨)이 음소 목록 상의 어느 음소에 해당하는지
        # s: 상태
        for t in range(1, feat_len):
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    # 자기 루프를 고려
                    self.alpha[l][s][t] = \
                        self.alpha[l][s][t - 1] \
                        + self.trans[p][s][0]
                    if s > 0:
                        # 선두(최초) 상태가 아니라 직전 상태에서의 전이를 고려
                        tmp = self.alpha[l][s - 1][t - 1] \
                              + self.trans[p][s - 1][1]
                        # 자기 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            self.alpha[l][s][t] = \
                                self.logadd(self.alpha[l][s][t],
                                            tmp)
                    elif l > 0:
                        # 선두(최초) 음소가 아니면서 선두 상태일 경우에는,
                        # 직전 음소의 마지막 상태에서 전이된 것이다
                        prev_p = label[l - 1]
                        tmp = self.alpha[l - 1][-1][t - 1] \
                              + self.trans[prev_p][-1][1]
                        # 자기 루프와 합을 계산
                        if tmp > self.LSMALL:
                            self.alpha[l][s][t] = \
                                self.logadd(self.alpha[l][s][t],
                                            tmp)
                    # else:
                    #   # 선두(최초) 음소이면서 선두 상태인 경우 자기 루프 외 전이는 불가능하다

                    # state_prob를 추가
                    self.alpha[l][s][t] += \
                        self.state_prob[l][s][t]

        # HMM 로그 빈도는 alpha 최종값
        self.loglikelihood = self.alpha[-1][-1][-1]

    def calc_beta(self, label):
        ''' 후향 확률 beta를 구한다
            left-to-right형의 HMM형을 전제로 구현한다
        label: 라벨
        '''
        # 라벨 길이와 프레임 수를 기록
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # alpha를 log(0)으로 초기화
        self.beta = self.LZERO * np.ones((label_len,
                                          self.num_states,
                                          feat_len))

        # t=-1 (최종 프레임)일 경우， 반드시 마지막 음소 최후 상태에 위치(확률은 log(1) = 0)
        self.beta[-1][-1][-1] = 0.0

        # t: 프레임
        # l: 라벨에서 몇 번째 음소인가
        # p: l(라벨)이 음소 목록상 어느 음소에 해당하는가
        # s: 상태
        # calc_alpha와 다르게，t는 feat_len-2에서 0으로 진행된다는 점에 주의
        for t in range(0, feat_len - 1)[::-1]:
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    # 자기 루프를 고려
                    self.beta[l][s][t] = \
                        self.beta[l][s][t + 1] \
                        + self.trans[p][s][0] \
                        + self.state_prob[l][s][t + 1]
                    if s < self.num_states - 1:
                        # 최종 상태가 아니라면 다음 상태로의 전이를 고려
                        tmp = self.beta[l][s + 1][t + 1] \
                              + self.trans[p][s][1] \
                              + self.state_prob[l][s + 1][t + 1]
                        # 자기 루프와 합을 계산
                        if tmp > self.LSMALL:
                            self.beta[l][s][t] = \
                                self.logadd(self.beta[l][s][t],
                                            tmp)
                    elif l < label_len - 1:
                        # 최종 음소가 아니면서 최종 상태인 경우에는 직후 음소 선두 상태로 전이
                        tmp = self.beta[l + 1][0][t + 1] \
                              + self.trans[p][s][1] \
                              + self.state_prob[l + 1][0][t + 1]
                        # 자기 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            self.beta[l][s][t] = \
                                self.logadd(self.beta[l][s][t],
                                            tmp)
                    # else:
                    # 최종 음소이며 최종 상태일 경우, 자기 루프 외 전이는 불가능

    def reset_accumulators(self):
        ''' accumulators (매개변수 갱신에 필요한 변수)를 초기화
        '''
        # GMM을 갱신하기 위한 accumulators
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

        # 천이 확률을 갱신하기 위한 accumulators
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
        ''' accumulators 갱신
            left-to-right를 전제로 한 구현이다
        feat: 특징
        label: 라벨
        '''
        # 라벨 길이를 기록
        label_len = len(label)
        # 프레임 수를 기록
        feat_len = np.shape(feat)[0]

        # t: 프레임    
        # l: 라벨상 몇 번째 음소인지
        # p: l(라벨)이 음소 목록상 어떤 음소인지
        # s: 상태
        for t in range(feat_len):
            for l in range(label_len):
                p = label[l]
                for s in range(self.num_states):
                    if t == 0 and l == 0 and s == 0:
                        # t=0 이면 반드시 선두 상태 (로그 확률이므로 log(1)=0)
                        lconst = 0
                    elif t == 0:
                        # t=0 이며 선두 상태가 아닌 경우는 확률이 0이므로 생략
                        continue
                    elif s > 0:
                        # t>0 이며 선두 상태가 아닌 경우 
                        # 자기 루프
                        lconst = self.alpha[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 작전 상태로부터의 전이를 고려
                        tmp = self.alpha[l][s - 1][t - 1] \
                              + self.trans[p][s - 1][1]
                        # 자기 루프와의 합을 계산
                        if tmp > self.LSMALL:
                            lconst = self.logadd(lconst,
                                                 tmp)
                    elif l > 0:
                        # t>0 선두 음소가 아니며 선두 상태인 경우
                        # 자기 루프
                        lconst = self.alpha[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 직전의 음소 마지막 상태에서 전이
                        prev_p = label[l - 1]
                        tmp = self.alpha[l - 1][-1][t - 1] \
                              + self.trans[prev_p][-1][1]
                        # 자기 루프와의 합 계산
                        if tmp > self.LSMALL:
                            lconst = self.logadd(lconst,
                                                 tmp)
                    else:
                        # 선두 음소이며 선두 상태일 경우 자기 루프만 가능
                        lconst = self.alpha[l][s][t - 1] \
                                 + self.trans[p][s][0]

                    # 후향 확률과 1/P를 추가
                    lconst += self.beta[l][s][t] \
                              - self.loglikelihood
                    # accumulators 갱신
                    for m in range(self.num_mixture):
                        pdf = self.pdf[p][s][m]
                        L = lconst \
                            + np.log(pdf['weight']) \
                            + self.elem_prob[l][s][m][t]

                        pdf_accum = self.pdf_accumulators[p][s][m]
                        # 평균값 벡터 갱신 수식 분자값은 로그를 취하지 않는다
                        pdf_accum['mu']['num'] += \
                            np.exp(L) * feat[t]
                        # 분모는 로그를 취한 후 갱신
                        if L > self.LSMALL:
                            pdf_accum['mu']['den'] = \
                                self.logadd(pdf_accum['mu']['den'],
                                            L)
                        # 대각-공분산 갱신 수식 분자값은 로그를 취하지 않는다
                        dev = feat[t] - pdf['mu']
                        pdf_accum['var']['num'] += \
                            np.exp(L) * (dev ** 2)
                        # 분모는 평균값과 동일한 값
                        pdf_accum['var']['den'] = \
                            pdf_accum['mu']['den']

                        # GMM 가중치 갱신 수식의 분자 값은 평균/분산 분모와 동일한 값
                        pdf_accum['weight']['num'] = \
                            pdf_accum['mu']['den']

        # 천이 확률 accumulators와 GMM 가중치 accumlators 분모 값을 갱신
        for t in range(feat_len):
            for l in range(label_len):
                p = label[l]
                for s in range(self.num_states):
                    # GMM 가중치 accumulator의 분모와
                    # 천이 확률 accumulator의 분모 갱신에 활용된다
                    alphabeta = self.alpha[l][s][t] \
                                + self.beta[l][s][t] \
                                - self.loglikelihood

                    # GMM 가중치 accumulator의 분모 갱신
                    for m in range(self.num_mixture):
                        pdf_accum = \
                            self.pdf_accumulators[p][s][m]
                        # 분모는 모든 m에 대해 동일한 값이므로 m == 0일 때만 계산
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

                    # 천이 확률 accumulator의 분모 값을 갱신
                    trans_accum = self.trans_accumulators[p][s]
                    if t < feat_len - 1 \
                            and alphabeta > self.LSMALL:
                        trans_accum['den'] = \
                            self.logadd(trans_accum['den'],
                                        alphabeta)

                    # 이하는 천이 확률 accumlator의 분자 값 갱신
                    if t == feat_len - 1:
                        # 최종 프레임은 생략
                        continue
                    elif s < self.num_states - 1:
                        # 각 음소가 마지막 상태가 아닌 경우 
                        # 자기 루프
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][0] \
                              + self.state_prob[l][s][t + 1] \
                              + self.beta[l][s][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)

                        # 전이
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
                        # 마지막 상태이면서 최종 음소가 아닌 경우
                        # 자기 루프
                        tmp = self.alpha[l][s][t] \
                              + self.trans[p][s][0] \
                              + self.state_prob[l][s][t + 1] \
                              + self.beta[l][s][t + 1] \
                              - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)
                        # 다음 음소의 시작 상태로 전이
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
                        # 자기 루프
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
        ''' 매개변수 갱신
        '''
        for p in range(self.num_phones):
            for s in range(self.num_states):
                # 천이 확률 갱신
                trans_accum = self.trans_accumulators[p][s]
                self.trans[p][s] = \
                    trans_accum['num'] - trans_accum['den']
                # 확률 합이 1이 되도록 정규화
                tmp = self.logadd(self.trans[p][s][0],
                                  self.trans[p][s][1])
                self.trans[p][s] -= tmp
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    pdf_accum = self.pdf_accumulators[p][s][m]
                    # 평균값 벡터 갱신
                    den = np.exp(pdf_accum['mu']['den'])
                    if den > 0:
                        pdf['mu'] = pdf_accum['mu']['num'] / den
                    # 대각-공분산 갱신
                    den = np.exp(pdf_accum['var']['den'])
                    if den > 0:
                        pdf['var'] = pdf_accum['var']['num'] / den
                    # 분산 플로어링
                    pdf['var'][pdf['var'] < self.MINVAR] = \
                        self.MINVAR
                    # gConst항 갱신
                    gconst = self.calc_gconst(pdf['var'])
                    pdf['gConst'] = gconst

                    # GMM 가중치 갱신
                    tmp = pdf_accum['weight']['num'] - \
                          pdf_accum['weight']['den']
                    pdf['weight'] = np.exp(tmp)
                # GMM 가중치 합이 1이 되도록 정규화
                wsum = 0.0
                for m in range(self.num_mixture):
                    wsum += self.pdf[p][s][m]['weight']
                for m in range(self.num_mixture):
                    self.pdf[p][s][m]['weight'] /= wsum

    # MonoPhoneHMM 클래스 음성인식 코드
    def viterbi_decoding(self, label):
        ''' 비터비 알고리즘에 의한 디코딩
            left-to-right HMM을 전제로 구현되어 있다
        lable: 라벨
        '''
        # 라벨 길이와 프레임 수를 구한다
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # score를 log(0)로 초기화
        self.score = self.LZERO * np.ones((label_len,
                                           self.num_states,
                                           feat_len))
        # Back Track용 천이 기록 영역
        # 0: 자기루프  1: 다음 상태로 전이
        self.track = np.zeros((label_len,
                               self.num_states,
                               feat_len), np.int16)
        # t = 0일 때, 반드시 최초 음소의 최초 상태다
        self.score[0][0][0] = self.state_prob[0][0][0]

        # t: 프레임
        # l: 라벨 위의 몇 번째 음소인가
        # p: l(라벨)이 음소 목록상의 어떤 음소인가
        # s: 상태
        for t in range(1, feat_len):
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    if s > 0:
                        # 선두 상태가 아닐 경우, 직전 상태에서 천이 혹은
                        # 자기 루프
                        p_next = self.score[l][s - 1][t - 1] \
                                 + self.trans[p][s - 1][1]
                        p_loop = self.score[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 확률이 높은 쪽을 선택
                        cand = [p_loop, p_next]
                        tran = np.argmax(cand)
                        self.score[l][s][t] = cand[tran]
                        self.track[l][s][t] = tran
                    elif l > 0:
                        # 선두 음성은 아니지만 선두 상태인 경우
                        # 직전의 음소 최종 상태로부터의 천이 혹은
                        # 자기 루프
                        prev_p = label[l - 1]
                        p_next = self.score[l - 1][-1][t - 1] \
                                 + self.trans[prev_p][-1][1]
                        p_loop = self.score[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        # 확률이 높은 쪽을 선택
                        cand = [p_loop, p_next]
                        tran = np.argmax(cand)
                        self.score[l][s][t] = cand[tran]
                        self.track[l][s][t] = tran
                    else:
                        # 선두 음소이면서 선두 상태일 경우 자기 루프만 가능하다
                        p_loop = self.score[l][s][t - 1] \
                                 + self.trans[p][s][0]
                        self.score[l][s][t] = p_loop
                        self.track[l][s][t] = 0

                    # state_prob를 추가
                    self.score[l][s][t] += \
                        self.state_prob[l][s][t]

        # 비터비 score는 마지막 scorer값
        self.viterbi_score = self.score[-1][-1][-1]

    def back_track(self):
        ''' 비터비 경로의 Back Track
        viterbi_path: Back Track 결과
        '''
        # 라벨 길이와 프레임 수를 구한다
        (label_len, _, feat_len) = np.shape(self.track)

        viterbi_path = []
        # 종료 지점에서부터 시작
        l = label_len - 1  # 음소
        s = self.num_states - 1  # 상태
        t = feat_len - 1  # 프레임
        while True:
            viterbi_path.append([l, s, t])
            # 시작 지점에 도달하면 종료
            if l == 0 and s == 0 and t == 0:
                break
            # trackの値を見る
            # 0이면 자기 루프, 1이면 천이
            tran = self.track[l][s][t]

            if tran == 1:
                # 천이
                if s == 0:
                    # 직전 음소로부터 천이
                    # l값을 줄이고 s를 종료 지점으로 한다
                    l = l - 1
                    s = self.num_states - 1
                else:
                    # 동일 음소의 직전 상태로부터 천이
                    # s값을 줄인다
                    s = s - 1
            # t값을 줄인다
            t = t - 1

        # viterbi_path를 역순으로 나열
        viterbi_path = viterbi_path[::-1]
        return viterbi_path

    # MonoPhoneHMM 클래스 혼합 수를 증가시키는 코드
    def mixup(self):
        ''' HMM 혼합 수를 2배 증가시킨다
        '''

        for p in range(self.num_phones):
            for s in range(self.num_states):
                pdf = self.pdf[p][s]
                for m in range(self.num_mixture):
                    # 혼합 가중치를 구한다
                    weight = pdf[m]['weight']
                    # 혼합 수를 2배 증가시켰으므로 가중치는 0.5
                    weight *= 0.5
                    # 복사본 혼합 가중치도 0.5배로 설정
                    pdf[m]['weight'] *= 0.5
                    # gConst 항을 구한다
                    gconst = pdf[m]['gConst']

                    # 평균값 벡터를 복사
                    mu = pdf[m]['mu'].copy()
                    # 대각 공분산을 복사
                    var = pdf[m]['var'].copy()

                    # 표준편차를 구한다
                    std = np.sqrt(var)
                    # 표준편차의 0.2배를 평균값 벡터에 더한다
                    mu = mu + 0.2 * std
                    # 원래 값의 평균값 벡터에서 0.2 * std를 뺀다
                    pdf[m]['mu'] = pdf[m]['mu'] - 0.2 * std

                    # 정규분포를 사전형으로 정의
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    # 정규분포를 추가
                    pdf.append(gaussian)

        # GMM 혼합 수를 2배로 한다
        self.num_mixture *= 2

    def train(self, feat_list, label_list, report_interval=10):
        ''' HMM을 1iteration만큼 갱신
        feat_list:  특징값 파일 목록
                    발화 ID가 key,특징값 파일 경로가 value인 사전
        label_list: 라벨 목록
                    발화 ID가 key，라벨이 value인 사전
                    辞書
        report_interval: 처리 중간 결과를 프린트하는 간격(발화 수 기준)
        '''
        # accumulators (매개변수 갱신에 사용되는 변수)를 reset
        self.reset_accumulators()

        # 특징값 파일을 하나씩 열어서 처리
        count = 0
        ll_per_utt = 0.0
        partial_ll = 0.0
        for utt, ff in feat_list.items():
            # 처리한 발화 수 카운트를 증가한다
            count += 1
            # 특징값 파일 열기
            feat = np.fromfile(ff, dtype=np.float32)
            # 프레임 수 x 차원 수 배열로 변형
            feat = feat.reshape(-1, self.num_dims)
            # 라벨값 추출
            label = label_list[utt]

            # 각 분포 출력 확률을 구한다
            self.calc_out_prob(feat, label)
            # 전향 확률을 구한다
            self.calc_alpha(label)
            # 후향 확률을 구한다
            self.calc_beta(label)
            # accumulators를 갱신한다
            self.update_accumulators(feat, label)
            # 로그 빈도를 더한다
            ll_per_utt += self.loglikelihood

            # 途中結果を表示する
            partial_ll += self.loglikelihood
            if count % report_interval == 0:
                partial_ll /= report_interval
                print('  %d / %d utterances processed' \
                      % (count, len(feat_list)))
                print('  log likelihood averaged' \
                      ' over %d utterances: %f' \
                      % (report_interval, partial_ll))

        # 모델 매개변수 갱신
        self.update_parameters()
        # 로그 빈도 발화 평균을 구한다
        ll_per_utt /= count
        print('average log likelihood: %f' % (ll_per_utt))

    def recognize(self, feat, lexicon):
        ''' 고립단어 인식을 수행한다
        feat:    특징값
        lexicon: 인식 대상 단어 목록
                 아래와 같은 사전형 데이터 리스트
                 {'word':단어, 
                  'pron':음소열,
                  'int':음소열 수치 표기}
        '''
        # 단어 목록 내의 단어별 빈도를 계산
        # 결과 목록
        result = []
        for lex in lexicon:
            # 음소 배열의 수치 표기를 구한다
            label = lex['int']
            # 각 분포의 출력 확률을 구한다
            self.calc_out_prob(feat, label)
            # 비터비 알고리즘을 실행
            self.viterbi_decoding(label)
            result.append({'word': lex['word'],
                           'score': self.viterbi_score})

        # 결과를 내림차순으로 정렬
        result = sorted(result,
                        key=lambda x: x['score'],
                        reverse=True)
        # 인식 결과를 반환
        return (result[0]['word'], result)

    def set_out_prob(self, prob, label):
        ''' 출력 확률을 설정하다
        prob DNN이 출력할 확률을 상정[프레임수 x (음소수*상태수)]의 2차원 배열로 되어 있다
        label 1 통화분 라벨
        '''
        # 프레임 수 파악
        feat_len = np.shape(prob)[0]
        # 라벨의 길이 파악
        label_len = len(label)

        # 각 상태(q,s)에서의 시각 t 출력 확률
        # (state_prob = sum(weight*elem_prob))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        # state_prob를 계산해 나가다
        # l: 라벨 위의 몇 번째 음소인가
        # p: l이 음소 리스트 상의 어느 음소인지
        # s: 상태
        # t: 프레임
        for l, p in enumerate(label):
            for s in range(self.num_states):
                # 음소 p의 상태 s 값은 DNN 출력상에서 p*num_states+s 에 저장되어 있는
                state = p * self.num_states + s
                for t in range(feat_len):
                    self.state_prob[l][s][t] = \
                        prob[t][state]

    def recognize_with_dnn(self, prob, lexicon):
        ''' DNN이 출력한 확률값을 이용하여 고립 단어 인식을 하다
        prob: DNN 출력 확률 (단, 각 상태의 사전 확률로 나누어 우도로 변환해 둘 것)
        lexicon: 인식 단어 목록.
                 아래의 사서형이 리스트에 나와 있다.
                {'word': 단어,
                'pron': 음소열,
                'int': 음소열 수치 표기 }
        '''
        # 단어 목록 내의 단어마다 우도를 계산하다
        # 결과 목록
        result = []
        for lex in lexicon:
            # 음소열의 수치 표기를 얻다
            label = lex['int']
            # 각 분포의 출력 확률을 세트하다
            self.set_out_prob(prob, label)
            # 비타비 알고리즘 실행
            self.viterbi_decoding(label)
            result.append({'word': lex['word'],
                           'score': self.viterbi_score})

        # 스코어의 승순으로 정렬하다
        result = sorted(result,
                        key=lambda x: x['score'],
                        reverse=True)
        # 인식 결과와 점수 정보를 반환
        return (result[0]['word'], result)

    def phone_alignment(self, feat, label):
        ''' 음소 얼라인먼트를 수행한다
        feat: 특징값
        label: 라벨
        '''
        # 각 분포의 출력 확률을 구한다
        self.calc_out_prob(feat, label)
        # 비터비 알고리즘 실행
        self.viterbi_decoding(label)
        # Back Track 실행
        viterbi_path = self.back_track()
        # 비터비 경로에서 프레임을 음소 배열로 변환
        phone_alignment = []
        for vp in viterbi_path:
            # 라벨 위의 음소 인덱스를 구한다
            l = vp[0]
            # 음소 번호를 음소 목록상의 번호로 변환
            p = label[l]
            # 번호를 음소 기호로 변환
            ph = self.phones[p]
            # phone_alignment 끝에 추가
            phone_alignment.append(ph)

        return phone_alignment

    def state_alignment(self, feat, label):
        ''' HMM 상태에서의 얼라인먼트를 실시하다
        feat: 특징량
        label: 라벨
        state_alignment: 프레임별 상태 번호
                    단, 여기서의 상태 번호는(음소번호)*(상태수)+(음소내상태번호)라고 하자.
        '''
        # 각 분포의 출력 확률을 구하다
        self.calc_out_prob(feat, label)
        # 비타비 알고리즘 실행
        self.viterbi_decoding(label)
        # Back Track 실행
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
            # 출력 시 상태 번호는
            # p * num_states + s 로 표시
            state = p * self.num_states + s
            # phone_ alignment 끝에 추가
            state_alignment.append(state)

        return state_alignment

    def save_hmm(self, filename):
        ''' HMM 파라미터를 json 형식으로 저장
        filename: 저장 파일명
        '''
        # json 형식으로 보존하기 위해
        # HMM의 정보를 사전 형식으로 변환하다
        hmmjson = {}
        # 기본 정보 입력
        hmmjson['num_phones'] = self.num_phones
        hmmjson['num_states'] = self.num_states
        hmmjson['num_mixture'] = self.num_mixture
        hmmjson['num_dims'] = self.num_dims
        # 음소 모형 리스트
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
                # hmm목록에 추가
                model_p['hmm'].append(model_s)
            # 음소 모델 리스트에 추가
            hmmjson['hmms'].append(model_p)

        # JSON 형식으로 보존
        with open(filename, mode='w') as f:
            json.dump(hmmjson, f, indent=4)

    def load_hmm(self, filename):
        ''' json 형식의 HMM 파일 가져오기
        filename: 읽기 파일명
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
        # 음소번호 p, 상태번호 s의 천이확률은 trans[p][s] = [loop, next]
        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                hmm = hmms['hmm'][s]
                # 遷移確率の読み込み
                tmp_trans = np.array(hmm['trans'])
                # 총합이 1이 되도록 정규화
                tmp_trans /= np.sum(tmp_trans)
                # 로그로 변환
                for i in [0, 1]:
                    tmp_trans[i] = np.log(tmp_trans[i]) \
                        if tmp_trans[i] > self.ZERO \
                        else self.LZERO
                tmp_p.append(tmp_trans)
            # self.transに追加
            self.trans.append(tmp_p)

        # 정규 분포 파라미터 가져오기
        # 음소번호 p, 상태번호 s, 혼합요소번호 m
        # 의 정규 분포는 pdf[p][s][m]로 액세스한다
        # pdf[p][s][m] = gaussian
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
                    # 정규 분포를 작성
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            # self.pdf에 추가
            self.pdf.append(tmp_p)