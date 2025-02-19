import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cupy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from CH7.MultiCNN import MultiCNN
from common.trainer import Trainer
import pickle as pkl

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, precision=np.float16)

# 드롭아웃 사용 유무와 비울 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.2
# ====================================================
 
network = MultiCNN(use_batch_norm=False, precision=np.float32)

file_name = 'init_params.pkl'
if not os.path.exists(file_name):
    with open(file_name, 'wb') as f:
        pkl.dump(network.params, f)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100, optimizer='adam', optimizer_param={'lr': 0.001},verbose=False , precision=np.float32)

trainer.train(5000)

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = range(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)  
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()