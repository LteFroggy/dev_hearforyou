'''
    데이터가 항상 머신러닝 알고리즘 학습에 필요한 형태로 제공되지는 않는다. 따라서 우리는 이걸 변형해줘야 함.
    모든 TorchVision 데이터셋은 변형 로직을 가지는, 호출 가능한 객체(Callable)를 받는 매개변수 두개를 갖는다.
    보통 torchvision.trasforms 모듈에서 주로 사용하는 변형을 제공해줌
    
    앞에서 사용한 FashionMNIST 특징은 PIL Image 형식이며, 정답은 정수이다. 학습을 위해서는 정규화된 텐서의 특징(feature)과 one-hot으로 부호화(encode)된 텐서 형태의 정답이 필요함.
    이러한 변형을 위해 ToTensor와 Lambda를 사용한다.
'''

import torch
import os
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"),
    train = True,
    download = True,
    transForm = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = torch.float).scatter_(0,torch.tensor(y), value=1))
)

# ToTensor는 PIL Image나 Numpy ndarray를 FloatTensor로 변환하고, 이미지의 픽셀의 크기(intensity)값을 [0., 1.]범위로 비례하요 조정(scale)한다.

# Lambda 변형은 사용자 정의 람다 함수를 적용한다. 여기서는 정수를 one-hot부호화된 텐서로 바꾸는 함수를 정의한다
# 위에서는 어떻게 된 걸까? torch.zeros(10)으로 10개짜리를 만들고, scatter_(in-place)를 사용한다.
# scatter는 값을 대체하는 함수인데, scatter_(dimension, index, src)의 형식을 가진다.
# dim이 0이니 열벌로 적용하는 것이고, 첫 번째 열의 y번째 값을 1로 바꾸는 함수이다.
# 따라서 one - hot encoding의 수행이 가능함.