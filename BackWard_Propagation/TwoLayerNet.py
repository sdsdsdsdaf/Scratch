import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from collections import OrderedDict

from common.Layers import *
from common.Gradient_pytorch import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, device='cpu'):

        self.device = device
        self.params = {}
        self.layers = OrderedDict()

        self.params['W1'] = weight_init_std * \
        torch.randn(input_size, hidden_size, device=self.device)

        self.params['b1'] = weight_init_std * \
        torch.zeros(hidden_size, device=self.device)

        self.params['W2'] = weight_init_std * \
        torch.randn(hidden_size, output_size, device=self.device)

        self.params['b2'] = weight_init_std * \
        torch.zeros(output_size, device=self.device)

        # Create Affine Layers
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'], device= self.device)
        self.layers['ReLu1'] = ReLu()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'], device = self.device)

        self.lastLayer = SoftmaxWithLoss(device=device)

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)

        return loss
    
    def accuracy(self, x, t):

        y = self.predict(x)

        y = torch.argmax(y, axis = 1)
        if t.ndim != 1:
            t = torch.argmax(t, axis = 1)

    def numerical_gradient(self, x, t):
        loss_W = lambda W :self.loss(x,t)

        grads ={}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'],self.device)
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'],self.device)
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'],self.device)
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'],self.device)

        return grads
    
    def gradient(self, x, t):

        #forward Propagation
        self.loss(x, t)

        #BackWard Propagation

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db

        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads