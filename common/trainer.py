# coding: utf-8
import os, sys
os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
#import numpy as np
import cupy as np
from common.optimizer import *
import time
from tqdm import tqdm
import pickle as pkl

class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True, precision = np.float64):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch #샘플로 평가 진행할 시
        self.precision = precision
        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        optimizer_param['precision'] = self.precision
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1) #1에폭당 실시할 학습의 횟수
        self.max_iter = int(epochs * self.iter_per_epoch) #총 학습 횟수
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.start_time = None #시가 측정시 필요하 코드드

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        # if self.current_iter % 100 == 0: # 학습 추이 파악을 위해 100번 학습당 정확도 평가
        if self.current_iter % self.iter_per_epoch == 0 and not self.current_iter == 0: #1에폭당 평가
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t] #t번째 데이터까지 잘라서 씀
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample, self.batch_size)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample, self.batch_size)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose and self.current_epoch % 10 == 0 and not self.current_epoch == 0 :
                spend_time = time.time() - self.start_time
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) +", time per epoch:" + str(round(spend_time, 4)) +" ===")
                self.start_time = time.time()
        self.current_iter += 1

    def save_acc_list(self, file_name='Acc_List.pkl'):
        save_file = {'test': self.test_acc_list, 'train': self.train_acc_list}
        with open(file_name, 'wb') as f:
            pkl.dump(save_file, f)
        
    def load_acc_list(self, file_name='Acc_List.pkl'):
        with open(file_name, 'rb') as f:
            save_file = pkl.load(f)
            self.test_acc_list = save_file['train']
            self.train_acc_list = save_file['test']

        print("Complete load Acc List")

    def train(self, frequency_of_save=None):
        if frequency_of_save is None:
            frequency_of_save = self.iter_per_epoch
        start = 0
        params_file_name = 'params.pkl'
        iter_num_file_name = 'iter_num.pkl'
        acc_list_file_name = 'Acc List.pkl'
        if os.path.exists(params_file_name):
            self.network.load_params(params_file_name)

        if os.path.exists(iter_num_file_name):
            with open(iter_num_file_name, 'rb') as f:
                start = pkl.load(f)
                print(f"Proceing Iter Num is {start}")

        if os.path.exists(acc_list_file_name):
            self.load_acc_list(acc_list_file_name)

        self.start_time = time.time()

        

        for idx in tqdm(range(0, self.max_iter), initial=start):
                
            if start + idx >= self.max_iter:
                break

            self.train_step()

            if (start + idx) % frequency_of_save == 0 and not idx == 0:
                self.network.save_params(params_file_name)
                self.save_acc_list(acc_list_file_name)
                
                with open(iter_num_file_name, 'wb') as f:
                    pkl.dump(start + idx, f)

        if start < self.max_iter:         
            self.save_acc_list(acc_list_file_name)
            self.network.save_params(params_file_name)
            with open(iter_num_file_name, 'wb') as f:
                    pkl.dump(start + idx, f)  #최종결과 저장

        test_acc = self.network.accuracy(self.x_test, self.t_test, self.batch_size)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

