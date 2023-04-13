# mean_std.py: 특징값 벡터 평균과 표준편차 계산하기

import os
import numpy as np

if __name__ == "__main__":
    # 특징값 2종류
    feature_list = ['fbank', 'mfcc']

    # 특징값 [fbank, mfcc] 각각에 대해 실행
    for feature in feature_list:
        # 각 특징 파일 목록과 평균 및 표준편차 계산 결과 저장 위치
        train_small_feat_scp = './%s/train_small/feats.scp' % (feature)
        train_small_out_dir = './%s/train_small' % (feature)
        train_large_feat_scp = './%s/train_large/feats.scp' % (feature)
        train_large_out_dir = './%s/train_large' % (feature)

        # 특징량 파일 리스트와 출력처를 리스트에 포함
        feat_scp_list = [train_small_feat_scp,
                         train_large_feat_scp]
        out_dir_list = [train_small_out_dir,
                        train_large_out_dir]
        # 각 세트에 대해 처리를 실행
        for (feat_scp, out_dir) in \
                zip(feat_scp_list, out_dir_list):
            print('Input feat_scp: %s' % (feat_scp))

            # 출력 디렉토리가 존재하지 않는 경우 작성
            os.makedirs(out_dir, exist_ok=True)

        # 특징값 평균과 분산
        feat_mean = None
        feat_var = None
        # 총 프레임 수
        total_frames = 0

        # 특징값 리스트 열기
        with open(feat_scp, mode='r') as file_feat:
            # 특징값 리스트를 1행씩 읽기
            for i, line in enumerate(file_feat):
                # 각 행에는 발화 ID, 특징값 파일 경로, 프레임 수, 차원 수가 스페이스로 구분되어 있다
                # split 함수를 사용해 스페이스로 구분된 행을 리스트 변수로 변환
                parts = line.split()
                # 0번째가 발화 ID
                utterance_id = parts[0]
                # 1번째가 특징값 파일 경로
                feat_path = parts[1]
                # 2번째가 프레임 수
                num_frames = int(parts[2])
                # 3번째가 차원 수
                num_dims = int(parts[3])

                # 특징값 데이터를 특징값 파일에서 읽기
                feature = np.fromfile(feat_path, dtype=np.float32)

                # 읽어온 시점에서 feature는 1행 벡터 (요소 수 = 프레임 수 * 차원 수)에 저장
                # 이를 프레임 수 x 차원 수의 행렬 형식으로 변환
                feature = feature.reshape(num_frames, num_dims)

                # 최초 파일을 처리했을 때 평균과 분산 초기화
                if i == 0:
                    feat_mean = np.zeros(num_dims, np.float32)
                    feat_var = np.zeros(num_dims, np.float32)

                # 총 프레임 수 더하기
                total_frames += num_frames
                # 특징값 벡터 프레임 전체 합을 더하기
                feat_mean += np.sum(feature, axis=0)
                # 특징값 벡터 제곱의 프레임 전체 합을 더하기
                feat_var += np.sum(np.power(feature, 2), axis=0)

        # 총 프레임 수로 나누어서 평균값 벡터 계산
        feat_mean /= total_frames
        # 분석값 벡터 계산
        feat_var = (feat_var / total_frames) - np.power(feat_mean, 2)
        # 제곱근을 취해서 표준편차 벡터 산출
        feat_std = np.sqrt(feat_var)

        # 파일에 쓰기
        out_file = os.path.join(out_dir, 'mean_std.txt')
        print('Output file: %s' % (out_file))
        with open(out_file, mode='w') as file_o:
            # 제곱근 벡터 쓰기
            file_o.write('mean\n')
            for i in range(np.size(feat_mean)):
                file_o.write('%e' % (feat_mean[i]))
            file_o.write('\n')
            # 표준편차 벡터 쓰기
            file_o.write('std\n')
            for i in range(np.size(feat_std)):
                file_o.write('%e ' % (feat_std[i]))
            file_o.write('\n')