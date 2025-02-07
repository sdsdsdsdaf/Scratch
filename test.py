import torch

arr = torch.randn(2,3, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu') )

print(arr)