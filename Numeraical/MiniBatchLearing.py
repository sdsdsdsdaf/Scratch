import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Mnist_dataset import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1
count = 0

iters_per_epoch = max(1, int(train_size / batch_size))

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in tqdm(range(iters_num)):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in network.params.keys():
        network.params[key] -= lr*grad[key]

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

    if i % iters_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        count += 1


x = np.array(range(iters_num))
epoch_num_arr = np.array(range(count))
plt.subplot(121)
plt.plot(x, train_loss_list, label="LOSS")
plt.legend()
plt.title("LOSS")

plt.subplot(122)
plt.plot(epoch_num_arr, train_acc_list, linestyle = "--",label = "Train acc")
plt.plot(epoch_num_arr, test_acc_list, label = "Test Acc")
plt.legend()
plt.title("ACCURACY")

plt.show()
