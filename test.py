import torch
import numpy as np
arr = torch.randn(2,3, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu') )

#print(arr)

arr1 = np.zeros(10)
arr2 = np.ones(3)

arr1[np.arange(1,4)] = arr2
print(arr1)