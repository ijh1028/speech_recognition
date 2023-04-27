# 02_train_dnn.py : DNN 학습을 수행하는 코드

# Pytorch에 필요한 모듈 불러오기
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

# 작성한 Dataset 클래스 가져오기
from my_dataset import SequenceDataset

# hmmfunc.py 에서 MonoPhoneHMM 클래스 가져오기
from hmmfunc import MonoPhoneHMM

# 직접 정의한 DNN 모델 불러오기
from my_model import MyDNN

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import shutil

if __name__ == "__main__":

    # 학습 데이터 특징값 리스트
    train_feat_scp = '../01compute_features/mfcc/train_small/feats.scp'
    # 학습 데이터 라벨(얼라인먼트) 파일 경로
    train_label_file = './exp/data/train_small/alignment'

    # 학습 데이터에서 계산된 특징값 평균/표준편차 파일 경로
    mean_std_file = '../01compute_features/mfcc/train_small/mean_std.txt'

    # 개발 데이터 특징값 목록
    dev_feat_scp = '../01compute_features/mfcc/dev/feats.scp'
    # 개발 데이터 라벨(얼라인먼트) 파일 경로
    dev_label_file = './exp/data/dev/alignment'

    # HMM 파일
    # HMM 파일은 음소와 상태 정보를 얻기 위해 사용
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # 학습 결과를 저장할 디렉토리 경로
    output_dir = os.path.join('exp', 'model_dnn')

    # MiniBatch에 포함된 발화 수
    batch_size = 5

    # 최대 에포크 수
    max_num_epoch = 60

    # 은닉층 Layer 개수
    num_layers = 4

    # 은닉층 차원 수
    hidden_dim = 1024

    # splice: 전후 n개 프레임 특징값을 결합
    # 차원수는 (splice*2+1)배가 된다
    splice = 5

    # 초기 학습률
    initial_learning_rate = 0.008

    # 학습률 감소 및 Early stopping 등을 판정하기 위한 시작할 에포크 수
    # (= 해당 에포크까지는 validation 결과와 상관없이 학습이 진행된다)
    lr_decay_start_epoch = 7

    # 학습률을 감쇠하는 비율
    # (감쇠후학습률<-현재학습률*lr_decay_factor)
    # 1.0 이상이면 감쇠시키지 않는다.
    lr_decay_factor = 0.5

    # Early stopping의 역치
    # 최저 손실값이 해당 에포크만큼 개선되지 않을 경우 학습 종료
    early_stop_threshold = 3

    # 출력 디렉토리가 존재하지 않는 경우 작성한다.
    os.makedirs(output_dir, exist_ok=True)

    # 설정을 사전 형식으로 하다
    config = {'num_layers': num_layers,
              'hidden_dim': hidden_dim,
              'splice': splice,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch,
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold}

    # 설정을 JSON 형식으로 저장
    conf_file = os.path.join(output_dir, 'config.json')
    with open(conf_file, mode='w') as f:
        json.dump(config, f, indent=4)

    # 특징값 평균 / 표준편차 파일을 읽어오기
    with open(mean_std_file, mode='r') as f:
        # 모든 행을 읽기
        lines = f.readlines()
        # 1행(0시작)가 평균값 벡터(mean), 3행은 표준편차벡터(std)
        mean_line = lines[1]
        std_line = lines[3]
        # 공백으로 구분하여 리스트로 변환
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy array로 변환
        feat_mean = np.array(feat_mean,
                             dtype=np.float32)
        feat_std = np.array(feat_std,
                            dtype=np.float32)
    # 평균/표준편차 파일을 복사
    shutil.copyfile(mean_std_file,
                    os.path.join(output_dir, 'mean_std.txt'))

    # 차원 수 정보를 얻기
    feat_dim = np.size(feat_mean)

    # DNN 출력층의 차원수를 얻기 위해서 HMM의 음소수와 상태수를 얻다
    # MonoPhoneHMM 클래스를 호출하다
    hmm = MonoPhoneHMM()
    # HMM 읽어오기
    hmm.load_hmm(hmm_file)
    # DNN 출력층 차원 수는 음소 수 x 상태 수
    dim_out = hmm.num_phones * hmm.num_states
    # Batch 데이터 생성할 때 빈 라벨 공간은 dim_out 보다 크거나 같은 값으로 채운다
    pad_index = dim_out

    # 신경망 모델을 만들다
    # 입력 특징량의 차원수는 feat_dim * (2*splice+1)
    dim_in = feat_dim * (2 * splice + 1)
    model = MyDNN(dim_in=dim_in,
                  dim_hidden=hidden_dim,
                  dim_out=dim_out,
                  num_layers=num_layers)
    print(model)

    # 옵티마이저 정의
    # momentum SGD(stochastic gradient descent) 사용
    optimizer = optim.SGD(model.parameters(),
                          lr=initial_learning_rate,
                          momentum=0.99)

    # 학습 데이터의 데이터셋 인스턴스를 생성
    # padding_index는 dim_out 보다 크거나 같은 값으로 설정
    train_dataset = SequenceDataset(train_feat_scp,
                                    train_label_file,
                                    feat_mean,
                                    feat_std,
                                    pad_index,
                                    splice)
    # 개발 데이터의 데이터셋 인스턴스를 생성
    dev_dataset = SequenceDataset(dev_feat_scp,
                                  dev_label_file,
                                  feat_mean,
                                  feat_std,
                                  pad_index,
                                  splice)

    # 학습 데이터의 Data Loader를 호출
    # 학습 데이터는 셔플하여 사용한다
    # (num_worker는 클수록 처리가 빨르지만 PC 사양에 따라 임계치가 존재한다.PC의 스펙에 따라 설정)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    # 개발 데이터의 DataLoader를 호출
    # 개발 데이터는 셔플하지 않는다
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    # 크로스 엔트로피 손실함수를 사용
    criterion = \
        nn.CrossEntropyLoss(ignore_index=pad_index)

    # CUDA를 사용할 수 있는 경우는 모델 파라미터를 GPU로, 그렇지 않으면 CPU에 배치한다
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # 모델을 훈련 모드로 설정
    model.train()

    # 훈련 데이터 처리와 개발 데이터 처리를 for loop로 간단하게 기술하기 위해서 사전형 데이터로 변환
    dataset_loader = {'train': train_loader,
                      'validation': dev_loader}

    # 각 history의 손실값과 오인율 출력
    loss_history = {'train': [],
                    'validation': []}
    error_history = {'train': [],
                     'validation': []}

    # 해당 코드에서는 validation 시 손실값이 가장 작았던 모델을 저장
    # 그러므로 가장 낮은 손실값을 갖는 모델과 에포크 수를 변수에 저장
    best_loss = -1
    best_model = None
    best_epoch = 0
    # Early stoppingフラグ．True일 경우 학습을 멈춘다
    early_stop_flag = False
    # Early stopping 판정을 위한 손실값의 최솟값이 개선되지 않는 에포크를 카운트한다
    counter_for_early_stop = 0

    # 로그 파일 준비
    log_file = open(os.path.join(output_dir,
                                 'log.txt'),
                    mode='w')
    log_file.write('epoch\ttrain loss\t' \
                   'train err\tvalid loss\tvalid err')

    # 에포크만큼 반복
    for epoch in range(max_num_epoch):
        # early stop flag가 true일 경우 학습을 멈춘다
        if early_stop_flag:
            print('    Early stopping.' \
                  ' (early_stop_threshold = %d)' \
                  % (early_stop_threshold))
            log_file.write('\n    Early stopping.' \
                           ' (early_stop_threshold = %d)' \
                           % (early_stop_threshold))
            break

        # 에포크 표시
        print('epoch %d/%d:' % (epoch + 1, max_num_epoch))
        log_file.write('\n%d\t' % (epoch + 1))

        # train phase와 validation phase를 교차 실행
        for phase in ['train', 'validation']:
            # 현재 에포크에서의 누적 손실값과 발화 수
            total_loss = 0
            total_utt = 0
            # 현재 에포크에서의 누적 인식 오류 문자수와 총 문자 수
            total_error = 0
            total_frames = 0

            # 학습 및 개발 Phase의 DataLoader에서 1개의 MiniBatch를 추출
            # MiniBatch 전체에 대해 처리할 때까지 반복
            # MiniBatch에 포함된 데이터는 음성 특징값, 라벨, 프레임 수, 라벨 길이, 발화ID
            for (features, labels, feat_len,
                 label_len, utt_ids) \
                    in dataset_loader[phase]:

                # CUDA를 사용할 수 있는 경우는 데이터를 GPU로, 그렇지 않으면 CPU에 배치한다
                features, labels = \
                    features.to(device), labels.to(device)

                # 기울기 reset
                optimizer.zero_grad()

                # 모델 순전달파(forward pass)
                outputs = model(features)

                # 이 시점에서 outputs는 [배지사이즈 프레임수 라벨수] 3차원 tensor
                # Cross Entropy Loss [샘플수, 라벨수]의 2차원 tensor를 필요로 하기에 view를 사용하여 변형
                b_size, f_size, _ = outputs.size()
                outputs = outputs.view(b_size * f_size,
                                       dim_out)
                # labels는 [배치사이즈, 프레임]의
                # 이차원 텐서
                # Cross Entropy Loss를 사용하기 위해서는
                # [샘플 수]의 1차원 텐서로 하다
                # 필요하기 때문에 .view를 사용하여 변형한다.
                # 1차원으로의 변형은 view(-1)로 좋다.
                # (view(b_size*f_size)도 좋아)
                labels = labels.view(-1)

                # 손실값을 계산
                loss = criterion(outputs, labels)

                # 학습 phase에서는 오차 역전달파로 모델 매개변수를 갱신
                if phase == 'train':
                    # 기울기를 계산
                    loss.backward()
                    # 옵티마이저로 매개변수를 갱신
                    optimizer.step()

                # 손실값을 누적
                total_loss += loss.item()
                # 처리된 발화 수를 카운트
                total_utt += b_size

                # 프레임 단위의 오인율을 계산하기 위해 모델 결과값을 기록
                _, hyp = torch.max(outputs, 1)
                # 라벨에서 pad_index로 채운 프레임을 제외
                hyp = hyp[labels != pad_index]
                ref = labels[labels != pad_index]
                # 추정 라벨과 정답 라벨이 일치하지 않는 프레임 수를 구한다
                error = (hyp != ref).sum()

                # 오류 프레임 수를 누적
                total_error += error
                # 총 프레임 수를 누적
                total_frames += len(ref)

            # 이 phase에서 단일 에포크 처리가 끝난다

            # 손실값의 누적치를 처리한 발화 수로 나눈다
            epoch_loss = total_loss / total_utt
            # 화면과 로그 파일로 출력
            print('    %s loss: %f' \
                  % (phase, epoch_loss))
            log_file.write('%.6f\t' % (epoch_loss))
            # 이력에 추가
            loss_history[phase].append(epoch_loss)

            # 총 오류 프레임 수를 총 프레임 수로 나누어 에러율을 계산
            epoch_error = 100.0 * total_error \
                          / total_frames
            # 화면과 로그 파일로 출력
            print('    %s error rate: %f %%' \
                  % (phase, epoch_error))
            log_file.write('%.6f\t' % (epoch_error))
            # 이력에 추가
            error_history[phase].append(epoch_error.cpu())

            # validation phase 처리
            if phase == 'validation':
                if epoch == 0 or best_loss > epoch_loss:
                    # 손실값 최저치가 갱신된 경우 모델을 저장
                    best_loss = epoch_loss
                    torch.save(model.state_dict(),
                               output_dir + '/best_model.pt')
                    best_epoch = epoch
                    # Early stopping 판정용 카운터를 reset한다
                    counter_for_early_stop = 0
                else:
                    # 최솟값을 갱신하지 못할 경우
                    if epoch + 1 >= lr_decay_start_epoch:
                        # lr_decay_start_epoch보다 높은 에포크일 경우
                        if counter_for_early_stop + 1 \
                                >= early_stop_threshold:
                            # 손실함수 최솟값이 지정된 기준값 동안 연속으로 개선되지 않았을 경우
                            # Early stopping flag를 True로 설정
                            early_stop_flag = True
                        else:
                            # Early stopping 조건에 도달하지 않은 경우는 학습률을 감소시켜 학습을 계속 진행
                            if lr_decay_factor < 1.0:
                                for i, param_group \
                                        in enumerate( \
                                        optimizer.param_groups):
                                    if i == 0:
                                        lr = param_group['lr']
                                        dlr = lr_decay_factor \
                                              * lr
                                        print('    (Decay ' \
                                              'learning rate:' \
                                              ' %f -> %f)' \
                                              % (lr, dlr))
                                        log_file.write( \
                                            '(Decay learning' \
                                            ' rate: %f -> %f)' \
                                            % (lr, dlr))
                                    param_group['lr'] = dlr
                            # Early stopping 판정용 카운트를 증가시킨다
                            counter_for_early_stop += 1

    # 전체 에포크 종료
    # 학습된 모델을 저장하고 로그를 작성하다
    print('---------------Summary' \
          '------------------')
    log_file.write('\n---------------Summary' \
                   '------------------\n')

    # 최종 에포크에서 모델을 저장
    torch.save(model.state_dict(),
               os.path.join(output_dir, 'final_model.pt'))
    print('Final epoch model -> %s/final_model.pt' \
          % (output_dir))
    log_file.write('Final epoch model ->' \
                   ' %s/final_model.pt\n' \
                   % (output_dir))

    # 최종 phase 정보
    for phase in ['train', 'validation']:
        # 최종 에포크 손실값 출력
        print('    %s loss: %f' \
              % (phase, loss_history[phase][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))
        # 최종 에포크 에러율 출력
        print('    %s error rate: %f %%' \
              % (phase, error_history[phase][-1]))
        log_file.write('    %s error rate: %f %%\n' \
                       % (phase, error_history[phase][-1]))

    # 베스트 에포크의 정보
    # (validation 손실이 최소였던 에폭시)
    print('Best epoch model (%d-th epoch)' \
          ' -> %s/best_model.pt' \
          % (best_epoch + 1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)' \
                   ' -> %s/best_model.pt\n' \
                   % (best_epoch + 1, output_dir))
    for phase in ['train', 'validation']:
        # 베스트 에포크 손실값 출력
        print('    %s loss: %f' \
              % (phase, loss_history[phase][best_epoch]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][best_epoch]))
        # 베스트 에포크 오인율 출력
        print('    %s error rate: %f %%' \
              % (phase, error_history[phase][best_epoch]))
        log_file.write('    %s error rate: %f %%\n' \
                       % (phase, error_history[phase][best_epoch]))

    # 손실값 이력(Learning Curve) 그래프로 저장
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(loss_history[phase],
                 label=phase + ' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir + '/loss.png')

    # 인식 오인율의 이력 그래프로 만들어 보존
    fig2 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(error_history[phase],
                 label=phase + ' error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    fig2.legend()
    fig2.savefig(output_dir + '/error.png')

    # 로그 파일을 닫다
    log_file.close()