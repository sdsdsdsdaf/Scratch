import numpy as np
from common.functions import *
from common.util import im2col, col2im

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out

        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None  #dx 즉 x의 증가량에 따른 기울기 변화는 이전 층의 역전파를 위해 구하고
        self.db = None  # dW db 와 같은 가중치의 변화에 따른 손실함수의 값을 구하는 것이 학습의 목적'

    def forward(self, x):

        self.original_x_shape = x.shape #다시 입력 모양대로 reshape하기 위해 저장
        x = x.reshape(x.shape[0], -1)  #데이터 한개 즉 288개의 픽셀을 한줄로 처리
        self.x = x

        out = np.dot(self.x, self.b) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None        
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size # t = [0,0,0,0,1,0,0,0,0] 처럼 원핫 인코딩 되었을때때

        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1 #t = 7같은 경우 이때 정답인 경우에서 1을 빼줘야함 -> 그래야 확률이 커질수록(_1에 가까워 질수록) 값이 작아짐짐
            dx = dx / batch_size

        return dx*dout
    
class Dropout:

    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x*self.mask

        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    
class BatchNormalization:

    def __init__(self, gamma, beta, momentum=0.9, running_mean = None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean #후에 테스트할때 쓰이는 이동평균값
        self.running_var = running_var
        
        self.batch_size = None
        self.xc = None #편차
        self.std = None #표준편차
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape

        if x.ndim != 2: #합성곱 신경망일때
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)
    
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape #데이터가 한개당 데이터가 한줄로 처처리되어 있음
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D) #이전 데이터기 없기에 0으로 초기화

        if train_flg:
            mu = x.mean(axis = 0)
            xc = x - mu
            var = np.mean(xc**2, axis = 0)
            std = np.sqrt(var = 10e-7)
            xn = xc / std # 표준정규분포로 전환 -> 각각의 편차를 표준분호로 나눔

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn - xn
            self.std = std
            self.running_mean = self.momentum * mu + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            xc = x - self.running_mean #현재까지 계산한한 이동 평균으로 계산 -> 이전 데이터 즉 이전 배치의 이미지까지 포함하기 위해
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        
        out = self.gamma * xn + self.beta #x = r*x + b -> 표준 정규분포를 이동시키기 위한 값들
        return out
    
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(self, dout)
        dx = dx.reshape(*self.input_shape)

        return dx
    
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
