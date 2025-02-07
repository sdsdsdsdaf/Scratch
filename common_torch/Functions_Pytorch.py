import torch

def sigmoid(a):
    return 1 / (1 + torch.exp(-a))

def softmax(a, device):
    if a.ndim == 2:
        x = a.T
        x = x - torch.max(x, dim=0)[0]
        y = torch.exp(x) / torch.sum(torch.exp(x), axis = 0)
        return y.T.type(torch.float64).to(device)

    a = a - torch.max(a)
    return torch.exp(a) / torch.sum(torch.exp(a))

def sum_squares_error(y, t):
    return 0.5 * torch.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size()[0])
        y = y.reshape(1, y.size()[0])

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size() == y.size():
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -torch.sum(torch.log(y[torch.arange(batch_size), t])) / batch_size



if __name__ == "__main__":

    t = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = torch.tensor([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])

    print(cross_entropy_error(y, t))