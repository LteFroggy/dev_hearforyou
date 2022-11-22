'''
    사용방법
        1. 폴더 안에 있는 SoundFile폴더에 판단하고자 하는 음원을 넣는다
        2. 해당 음원파일의 이름을 main.py 파일 내부의 FILENAME 변수에 넣어준다
        3. 실행하면 결과가 출력된다.

    주의사항
        음원파일의 마지막 1초 부분만 떼어내어 판단에 활용하므로 마지막 부분에 소리가 없다면 판단이 잘 안될 수 있음
        Model.Analyze 함수의 ereaseFlag를 True로 설정하면 사용한 음원파일은 삭제된다.
        현재 분류할 수 있는 음원의 종류가 몇 개 없으며 정확도도 떨어질 수 있다.
        분류 가능한 음원의 종류는 폴더 안의 settings.py 파일에 labels라는 변수로 들어있음.
'''
import Model
import settings as set
from NeuralNetwork import NeuralNetwork

FILENAME = "사이렌 예시.m4a"

if __name__ == "__main__" :
    label = set.labels
    result = Model.Analyze(FILENAME, ereaseFlag = False)
    
    if result != -1 :
        print(f"{FILENAME}은 {label[result]} 입니다.")
    else :
        print(f"{FILENAME}이 무엇인지 확인하지 못했습니다")