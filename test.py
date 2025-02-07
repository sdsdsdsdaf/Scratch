import torch

arr = torch.randn(2,3, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu') )

li = [1,2,3,4,5]
li.reverse()

print(li)