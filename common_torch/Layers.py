import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
from common.Functions_Pytorch import *

class AddLayer:

    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out
    def backward(self, dout):
        dx = dout*1
        dy = dout*1

        return (dx,dy)
    
class ReLu:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.clone()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx= dout

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out
    
    def backward(self, dout):
        return dout * self.out * (1.0 -self.out)
    
class Affine:
    def __init__(self, W, b, device):
        self.W = W
        self.b = b
        self.device = device

        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = torch.matmul(self.x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = torch.matmul(dout, self.W.T.type(torch.float64).to(self.device))
        self.dW = torch.matmul(self.x.T.type(torch.float64).to(self.device), dout)
        self.db = torch.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape)
        return dx

class SoftmaxWithLoss:
    def __init__(self, device):
        self.loss = None
        self.y = None
        self.t = None
        self.device = device

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x, self.device)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size * dout

        return dx
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigmoid_layer = Sigmoid()

    test_arr = torch.randn(5,3, device=device)

    print(test_arr.shape[1])