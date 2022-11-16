'''
    Tensor에 대하여 학습한다.
    Tensor는 배열, 행렬과 굉장히 유사한 자료구조로, Pytorch에서는 이를 이용하여 모델의 입력, 출력 및 매개변수의 부호화(Encode)가 가능하다.

    Numpy와 ndarray와 유사하나, GPU나 하드웨어 가속기에서 실행 가능하다는 점이 차이이다.
    또한 자동 미분에 최적화되어있다.
'''

import torch
import numpy as np

# 텐서의 초기화 방법 1, 직접(directly) 데이터로부터 생성
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
    
# 텐서의 초기화 방법 2, NumPy 배열로부터 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 텐서의 초기화 방법 3, 다른 텐서로부터 생성(명시적으로 override하지 않으면, 인자로 주어진 텐서의 shape, datatype등의 속성을 유지한다)
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor : \n {x_ones} \n")

# 이 경우는 명시적으로 형태를 변환한 것, 앞에 있는 것은 Shape를 정하는 부분.
x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Rand Tensor : \n {x_rand} \n")

# 무작위 값 혹은 상수 값을 초기화에 사용하기
# Shape은 Tensor의 Dimension을 나타내는 tuple이다. 아래처럼 사용 가능하다
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# 동일 간격으로 생성하기. linspace(start, end, steps) 를 사용한다. steps는 몇 개를 생성할지를 정한다
lin_tensor = torch.linspace(2, 500, 200)
lin_tensor.reshape(8, 25)
print(lin_tensor, lin_tensor.shape)

# 텐서의 Attribute에 대한 내용. shape 및 datatype, 어느 장치에 저장되는지 등이 attribute에 속한다.
tensor = torch.rand(3,4)
print(f"Shape of Tensor : \n {tensor.shape} \n")
print(f"Datatype of tensor : \n {tensor.dtype} \n")
print(f"Device tensor is stored on : \n {tensor.device} \n" )

# 텐서 연산(Operation)
# 텐서는 전치, 인덱싱, 슬라이싱, 수학적 계산, 선형대수 연산, 임의 샘플링(random sampling)등이 모두 가능하다
# 모든 연산은 GPU에서 실행 가능하며, 기본적으로 텐서는 CPU에 생성되지만 tensor.to 메소드를 이용하여 명시적으로 이동이 가능하다.
# 너무 큰 텐서를 장치들 간에 복사하면 오버헤드가 클 수 있으므로 주의 
# 관련 연산 확인을 위해서는 https://pytorch.org/docs/stable/torch.html 를 확인하자.

# GPU가 존재하면 GPU로 텐서를 이동
if torch.cuda.is_available() :
    tensor = tensor.to("cuda")

# 연산 사용해보기
tensor = torch.ones(4,4)
tensor = tensor.type(torch.int32)
print(f"First row : {tensor[0]}")
print(f"First Column : {tensor[:, 0]}")
print(f"Last Column : {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
tensor = tensor.type(torch.float32)

# 텐서 합치기
# torch.cat을 사용하여 주어진 차원에 따라 일련의 텐서 연결 가능, 비슷하나 미묘하게 다른 기능을 수행하는 torch.stack도 참고해보자
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

# 산술 연산
# 두 텐서 간의 행렬 곱 구하기, y1, y2, y3는 모두 같은 값을 가진다
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)
print(y1)

# 요소별 곱 계산하기, 역시 z1, z2, z3는 같은 값을 가진다
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)
print(z1)

# 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서가 된다면, 단일 요소 텐서라고 하며, item()을 사용하여 Python 숫자 값으로 변환이 가능하다.
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# 바꿔치기(in-place) 연산
# 연산 결과를 피연산자에 저장하는 연산을 바꿔치기 연산이라고 하며, _의 접미사를 가진다. 예를 들어 x.copy_(y)나 x.t_()는 x를 변경하게 된다.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

