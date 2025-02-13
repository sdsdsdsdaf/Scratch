import torch
import cupy as np
import time
from collections import OrderedDict
arr = torch.randn(2,3, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu') )

#print(arr)

arr1 = np.zeros(10)
arr2 = np.ones(3)

arr1[np.arange(1,4)] = arr2
#print(arr1)

def test_func():
    time.sleep(1)

    
if __name__ == "__main__":
    layer_list = ['CONV', 'relu', 'pooling', 'affine', 'Relu', 'Affine', 'softmax']

    layer_list = [layer.lower() for layer in layer_list ]
    

    li = [1,2,3,4]
    dic = {}
    for idx, ele in enumerate(li):
        dic[f'W{idx}'] = ele

    mydic = OrderedDict()
    mydic.update({'1': 1, '2':2})

    key = input('입력하세요')
    key = key.lower()
    print(key in ('dropout', 'batchnorm'))