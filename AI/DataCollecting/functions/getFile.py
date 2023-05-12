import os
import pickle
from tqdm import tqdm
from pytube import YouTube
from pathlib import Path

'''
    Youtube 객체에서 가져올 수 있는 정보
    
    title           제목
    length          영상 길이
    author          게시자
    publish_date    게시날짜
    views           조회수
    keywords        키워드
    description     설명
    thumbnail_url   썸네일


    Youtube 객체를 .streams 해줌으로써 스트림 객체로 만들어냄
    사용 시에 .stream.filter를 이용해서 필터링도 가능
'''

def main(soundName) :
    print("파일 다운로드 및 라벨 수정 시작")
    basePath = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "data")
    basePath = os.path.join(basePath, soundName)
    savePath = os.path.join(basePath, "downloaded")
    pklPath = os.path.join(basePath, "summary.pkl")
    newPklPath = os.path.join(basePath, "name_label.pkl")

    if not (os.path.isdir(savePath)) :
        os.mkdir(savePath)

    with open(pklPath, "rb") as file : 
        labels = pickle.load(file)

    new_labels = {}
    basicLink = "https://www.youtube.com/watch?v="

    for line in tqdm(labels, desc = f"파일 다운로드 중") :
        link = basicLink + str(line[0])
        yt = YouTube(link)

        try :
            yt.streams.filter(only_audio = True).first().download(savePath)
            new_labels[yt.title] = [line[1], line[2]]
        except Exception as e :
            continue

    with open(newPklPath, "wb") as file :
        pickle.dump(new_labels, file)