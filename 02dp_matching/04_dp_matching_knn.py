# 04_dp_matching_knn.py : DP Matching과 템플릿 Matching을 사용하여 발화 인식을 수행

import numpy as np
import sys


def dp_matching(feature_1, feature_2):
    ''' DP Mathing을 수행한다
    입력:
        feature_1 : 비교할 특징값 배열 1
        feature_2 : 비교할 특징값 배열 2
    출력:
        total_cost : 최단 경로의 총 비용
        min_path :   최단 경로 프레임 대응
    '''
    # 프레임 수와 차원 수를 추출
    (nframes_1, num_dims) = np.shape(feature_1)
    nframes_2 = np.shape(feature_2)[0]

    # 거리(비용) 행렬을 계산
    distance = np.zeros((nframes_1, nframes_2))
    for n in range(nframes_1):
        for m in range(nframes_2):
            # feature_1의 n번째 프레임과 feature_2의 m번째 프레임의 유클리드 거리 제곱을 계산
            distance[n, m] = \
                np.sum((feature_1[n] - feature_2[m]) ** 2)

    # 누적 비용 행렬
    cost = np.zeros((nframes_1, nframes_2))
    # 이동 종류 (세로 / 사선 / 가로)를 기록하는 행렬
    # 0 : 세로 이동, 1 : 사선 이동, 2 : 가로 이동
    track = np.zeros((nframes_1, nframes_2), np.int16)

    # 시작 지점의 거리
    cost[0, 0] = distance[0, 0]

    # 0번째 행 : 반드시 세로로 (아래로) 이동한다
    for n in range(1, nframes_1):
        cost[n, 0] = cost[n - 1, 0] + distance[n, 0]
        track[n, 0] = 0

    # 0번째 열 : 반드시 가로로 (우측으로) 이동한다
    for m in range(1, nframes_2):
        cost[0, m] = cost[0, m - 1] + distance[0, m]
        track[0, m] = 2

    # 그외 : 가로, 세로, 사선중에서 최소 비용으로 이동한다
    for n in range(1, nframes_1):
        for m in range(1, nframes_2):
            # 세로로 이동했을 때 누적 비용
            vertical = cost[n - 1, m] + distance[n, m]
            # 사선으로 이동했을 때 누적 비용(사선은 2배 가중치를 부여한다)
            diagonal = cost[n - 1, m - 1] + 2 * distance[n, m]
            # 가로로 이동했을 때 누적 비용
            horizontal = cost[n, m - 1] + distance[n, m]

            # 누적 비용이 최소인 이동 경로를 선택
            candidate = [vertical, diagonal, horizontal]
            transition = np.argmin(candidate)

            # 누적 비용과 이동 방향을 기록
            cost[n, m] = candidate[transition]
            track[n, m] = transition

    # 총 비용은 cost 행렬의 최종행 x 최종열 값
    # 특징값의 프레임 수로 정규화
    total_cost = cost[-1, -1] / (nframes_1 + nframes_2)

    # Back Track 끝에서 track 값을 기준으로 역추적하여 최소 비용 경로를 구한다
    min_path = []
    # 최종 행 x 최종 열에서 시작
    n = nframes_1 - 1
    m = nframes_2 - 1
    while True:
        # 현재 프레임 위치를 min_path에 추가
        min_path.append([n, m])

        # 시작 지점에 도달하면 종료
        if n == 0 and m == 0:
            break

        # track 값을 확인
        if track[n, m] == 0:
            # 세로 이동일 경우
            n -= 1
        elif track[n, m] == 1:
            # 사선 이동일 경우
            n -= 1
            m -= 1
        else:
            # 가로 이동일 경우
            m -= 1

    # min_path를 역순으로 교체
    min_path = min_path[::-1]

    # 총 비용과 최종 경로를 출력
    return total_cost, min_path


#
# メイン関数
#
if __name__ == "__main__":
    # 認識対象のセット番号と発話番号
    query_set = 1
    query_utt = 9

    # K-nearest neighborのパラメータ
    K = 3

    # MFCCの次元数
    num_dims = 13
    # 総セット数
    num_set = 5
    # 発話の種類数
    num_utt = 10

    # 特徴量データを特徴量ファイルから読み込む
    query_file = './mfcc/REPEAT500_set%d_%03d.bin' % \
                 (query_set, query_utt)
    query = np.fromfile(query_file, dtype=np.float32)
    query = query.reshape(-1, num_dims)

    cost = []
    for set_id in range(1, num_set + 1):
        for utt_id in range(1, num_utt + 1):
            # query と同じセットは比較しない
            if set_id == query_set:
                continue

            # 比較対象の特徴量を読み込む
            target_file = './mfcc/REPEAT500_set%d_%03d.bin' % \
                          (set_id, utt_id)
            print(target_file)
            target = np.fromfile(target_file,
                                 dtype=np.float32)
            target = target.reshape(-1, num_dims)

            # DPマッチング実施
            tmp_cost, tmp_path = dp_matching(query, target)

            cost.append({'utt': utt_id,
                         'set': set_id,
                         'cost': tmp_cost
                         })

    # コストの昇順に並び替える
    cost = sorted(cost, key=lambda x: x['cost'])

    # コストのランキングを表示する
    for n in range(len(cost)):
        print('%d: utt: %d,  set: %d, cost: %.3f' % \
              (n + 1,
               cost[n]['utt'],
               cost[n]['set'],
               cost[n]['cost']))

    #
    # K-nearest neighbor を行う
    #
    voting = np.zeros(num_utt, np.int16)
    for n in range(K):
        # トップK個の発話IDで投票を行う
        voting[cost[n]['utt'] - 1] += 1

    # 投票の最も大きかった発話IDを出力する
    max_voted = np.argmax(voting) + 1
    print('Estimated utterance id = %d' % max_voted)