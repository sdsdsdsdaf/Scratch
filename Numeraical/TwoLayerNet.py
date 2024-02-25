import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
 
import numpy as np
from Functions import sigmoid, softmax, cross_entropy_error
from Grdient import numerical_gradient

#sys.path.append(os.pardir)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        
        self.params = {}
        self.params['W1'] = weight_init_std *\
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*\
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        acc = np.sum(y==t) / float(x.shape[0])

        return 
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        for weight_name, weight_value in self.params.items():
            grads[weight_name] = np.zeros_like(weight_value)

        for weight_name in self.params.keys():
            grads[weight_name] = numerical_gradient(loss_W, self.params[weight_name])

        return grads



if __name__ == "__main__":
    print("정상실행")