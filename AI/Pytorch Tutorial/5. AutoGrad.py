'''
    신경망의 학습에 가장 자주 사용되는 알고리즘은 역전파(Back-Propagation)이다.
    역전파 알고리즘에서 매개변수는 손실 함수의 Gradient에 따라 조정된다.
    변화도 계산을 위해서는 torch.autograd라는 자동 미분 엔진을 사용한다.
'''
import torch

# 입력 x, 매개변수 w와 b, Loss Function이 존재하는 단일 계층 신경망을 Pytorch에서 정의해보자
x = torch.ones(5) # input
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad = True) # 5개의 Input에서 3개의 Output으로 가야 하므로 5x3 개의 Weight
b = torch.randn(3, requires_grad= True) # 3개의 Output 뉴런이 5개의 Input에서 각각 값을 받은 뒤에 bias를 더해야 하므로 3개
z = torch.matmul(x, w) + b # 1x5, 5x3짜리 행렬의 곱이므로 1x3의 결과가 나오며, 1 x 3의 크기를 가지는 b를 각각 더하니 나오는 값은 실제로 모델이 내놓은 Output이 된다.
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 신경망에서, 최적화를 해야 할 대상인 매개변수는 w와 b이다. 따라서 이러한 변수들이 에러에 미치는 영향을 계산하고, 그를 바탕으로 값을 수정해내야 한다.
# 이를 위해 적용하는 특징이 requires_grad이다.
# 이는 나중에 w.requires_grad_(True)등을 통해 변경 역시 가능하다.
# 연산 그래프의 구성을 위해 텐서에 적용될 함수는 사실 Function 클래스의 객체이다.
# 해당 객체는 순전파(Forward-Propagation)방향의 함수 계산 방법과 역전파(Back-Propagation)단계에서의 도함수(derivative)를 계산하는 방법을 안다.
print(f"Gradient Function for z = {z.grad_fn}")
print(f"Gradient Function for loss = {loss.grad_fn}")

# 변화도(Gradient) 계산하기
# 신경망에서 매개변수의 가중치 최적화를 위해서는 매개변수에 대한 손실함수의 도함수를 계산해야 한다.
# 즉, x와 y의 일부 고정값에서 d(loss) / d(w), d(loss) / d(b) 가 필요한 것이다.
# 이러한 도함수의 계산을 위해 loss.backward()를 호출한 다음 w.grad와 b.grad에서 값을 가져온다. 해당 계산의 수행을 위해서는 requires_grad 가 True여야 한다.
# 주어진 그래프에서 backward를 사용한 변화도 계산은 한 번만 수행 가능하다. 만약 동일한 그래프에서 여러번의 backward를 호출해야 하면 retrain_graph = True가 전달되어야 함.
loss.backward()
print(w.grad)
print(b.grad)

# 변화도 추적 멈추기
# 기본적으로, requires_grad = True인 모든 텐서는 연산 기록을 추적하고 변화도 계산을 지원한다.
# 그러나 모델의 학습 사용시 등 입력 데이터를 단순히 적용하기만 하는 등의 Forward-Propagation만 필요한 경우에는 torch.no_grad()로 둘러싸서 추적을 멈출 수 있다.
z = torch.matmul(x, w) + b
print(z.requires_grad)
with torch.no_grad() :
    z = torch.matmul(x, w) + b
    print(z.requires_grad)