import numpy as np
import torch

def numerical_gradient(f, x, device = torch.device('cpu')):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x.to('cpu').numpy())

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    i=0
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()
        i+=1

    return torch.tensor(grad).type(torch.float64).to(device=device)