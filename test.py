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
    y = np.zeros((1,5))
    b = np.array([[2,4,5,6,7]])

    c = np.concatenate((y, b), 0)
    c = np.delete(c,0, 0)
    li = [1,2,3,4]
    dic = {}
    for idx, ele in enumerate(li):
        dic[f'W{idx}'] = ele

    mydic = OrderedDict()
    mydic.update({'1': 1, '2':2})

    for d in mydic:
        print(d)


from tqdm import tqdm
import pickle as pkl
import time
file_name = 'test.pkl'

tqdm