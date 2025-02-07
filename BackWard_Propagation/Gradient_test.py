import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from Mnist_dataset import load_mnist
from TwoLayerNet import TwoLayerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = torch.tensor(x_train).to(device=device)
t_train = torch.tensor(t_train).to(device=device)
x_test = torch.tensor(x_test).to(device=device)
t_test = torch.tensor(t_test).to(device=device)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, device = device)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = torch.mean( torch.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

    #print(torch.abs(grad_backprop[key] - grad_numerical[key]))