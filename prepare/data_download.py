# -*- coding: utf-8 -*-

# 이 책에서 다루는 음성 데이터와 라벨 데이터를 다운로드
# 데이터는 JSUT 코퍼스를 사용
# https://sites.google.com/site/shinnosuketakamichi/publication/jsut

# 파일 다운로드
from urllib.request import urlretrieve

# zip 파일
import zipfile

# os
import os

# 메인 함수
if __name__ == "__main__":
    
    # 데이터 저장 위치
    data_dir = 'D:/github/speech_recognition/data/original'

    # 디렉토리 data_dir가 존재하지 않으면 작성
    os.makedirs(data_dir, exist_ok=True)

    # 음성 파일(jsut 코퍼스.zip형식) 다운로드
    data_archive = os.path.join(data_dir, 'jsut-data.zip')
    print('download jsut-data start')
    urlretrieve('http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip', 
                data_archive)
    print('download jsut-data finished')

    # 다운로드한 zip 데이터 압축해제
    print('extract jsut-data start')
    with zipfile.ZipFile(data_archive) as data_zip:
        data_zip.extractall(data_dir)
    print('extract jsut-data finished')

    # zip 파일을 삭제
    os.remove(data_archive)

    # jsut 코퍼스 라벨 데이터 다운로드
    label_archive = os.path.join(data_dir, 'jsut-label.zip')
    print('download jsut-label start')
    urlretrieve('https://github.com/sarulab-speech/jsut-label/archive/master.zip',
                label_archive)
    print('download jsut-label finished')

    # 다운로드한 zip 데이터 압축해제
    print('extract jsut-label start')
    with zipfile.ZipFile(label_archive) as label_zip:
        label_zip.extractall(data_dir)
    print('extract jsut-label finished')

    # zip 파일을 삭제
    os.remove(label_archive)

    print('all processes finished')