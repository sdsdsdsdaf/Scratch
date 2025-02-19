import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
#import numpy as np
import cupy as np  
from collections import OrderedDict
from common.layers import *
import pickle
from collections import deque

FN, C,FH, FW, PAD, STRIDE = 0, 1, 2, 3, 4, 5
PH, PW, P_STRIDE = 0, 1, 2

class MultiCNN: #CNN으로 시작을 하지 않으면 이미지가 흐름대로 흘러가지 않음
    """합성곱 다층 신경망(확장판)
    가중치 감소, 드롭아웃, 배치 정규화 구현

    Parameters
    ----------
    input_dim : 입력 크기（MNIST의 경우엔 데니터 1개 일시 (1, 28,28) 체널, 높이, 너비
    layer_list : 각 층이 어떤계층인지 받는 인자
    filter_shape_list : 각 합성곱층의 필터 모양 (필터 개수, 필터높이, 필터 너비, 패딩, stride) #채널은 필요 없음 -> 나중에 dx 구할 때 쓰기 때문에 가중치(필터) 모양대로만 만듬 ->2차원 리스트
    hidden_layer_list affine계층의 은닉층 -> 2차원 리스트
    pooling_list = 각 풀링층의 모양 (fh, hw, pad) -> 2차원 리스트
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기
    use_dropout : 드롭아웃 사용 여부
    dropout_ration : 드롭아웃 비율
    use_batchNorm : 배치 정규화 사용 여부
    """
    def __init__(self, input_dim = (1, 28, 28), layer_list = ['conv', 'relu', 'pooling', 'affine', 'relu', 'affine', 'softmax'], #여기서 relu, softmax, pooling은 가중치가 없는 계층
                 filter_shape_list = [[30, 1, 5, 5, 0, 1]], hidden_layer_list = [100], pooling_list = [[2, 2, 2], [2,2,2]], output_size = 10, activation = "ReLu", weight_init_std = 'ReLu', weight_decay_lambda=0,
                 use_dropout = False, dropout_ration = 0.5, use_batch_norm=False, precision = np.float64):
        layer_list = [layer.lower() for layer in layer_list ]
        try:
            assert layer_list.count('conv') == len(filter_shape_list), "Not match Convulution Layer`s numer and filter layer Number"
            assert layer_list.count('affine') == len(hidden_layer_list) + 1, "Not match Hidden Layer`s number and hidden layer list num"
            assert layer_list[-1] in ('softmax', 'identity'), "Output Layer`s activation function must be 'Softmax' or 'Identity function'"
            assert activation.lower() in ('relu', 'sigmoid'), "Not implement exclude relu and sigmoid activation function" 
            assert weight_init_std.lower() in ('relu', 'sigmoid', 'he', 'xavier'), "Not implement exclude relu and sigmoid activation function" 

            for filter_shape in filter_shape_list:
                assert len(filter_shape) == 6, "You have write 6 element (FN, C, FH, FW, PAD, STRIDE)"
        except:
            sys.exit()

        if not np.array(filter_shape_list).ndim == 2:
            filter_shape_list =[filter_shape_list] 
        if not np.array(pooling_list).ndim == 2:
            pooling_list = [pooling_list]

        self.use_batchnorm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_ration = dropout_ration
        self.weight_decay_lambda = weight_decay_lambda
        self.precision = precision
        self.output_size = output_size

        conv_parm = []
        for filter_shape in filter_shape_list:
            conv_parm.append({'fliter_num': filter_shape[FN], 'filter_ch' : filter_shape[C],'fliter_h': filter_shape[FH], 'fliter_w': filter_shape[FW], 'pad': filter_shape[PAD], 'stride': filter_shape[STRIDE]})
        
        # 'he' 초깃값 활용 위해 앞 노드 개수 계산 conv계층은 필터 한개의 크기 c*fh*fw, affine계층은 앞 노드의 출력 노드 개수
        pre_node_num = [] 
        fc = input_dim[-3]
        out_h = input_dim[-2]
        out_w = input_dim[-1]
        prev_node = 0
        conv_i = 0
        affine_i = 0
        pooling_i = 0
        output_node_num = deque()
        
        hidden_layer_list.append(output_size)

        for layer in layer_list:
            if layer == 'conv':
                prev_node = fc * filter_shape_list[conv_i][FH] * filter_shape_list[conv_i][FW] 
                pre_node_num.append(prev_node) #cnn에서는 이미지 블록 한개의 크기  --> 가중치의 개수
                

                out_h = (out_h + 2*filter_shape_list[conv_i][PAD] - filter_shape_list[conv_i][FH]) // filter_shape_list[conv_i][STRIDE] + 1  
                out_w = (out_w + 2*filter_shape_list[conv_i][PAD] - filter_shape_list[conv_i][FW]) // filter_shape_list[conv_i][STRIDE] + 1
                fc = filter_shape_list[conv_i][FN]
                output_node_num.append(fc * out_h * out_w) #전체 노드의 개수
                conv_i += 1
            
            if layer == 'pooling':
                out_h = (out_h - pooling_list[pooling_i][PH]) // pooling_list[pooling_i][P_STRIDE] + 1
                out_w = (out_w - pooling_list[pooling_i][PW]) // pooling_list[pooling_i][P_STRIDE] + 1
                prev_node = fc * out_h * out_w
                pre_node_num.append(prev_node)
                pooling_i += 1
                

            if layer == 'affine':
                pre_node_num.append(prev_node)
                prev_node = hidden_layer_list[affine_i]
                output_node_num.append(hidden_layer_list[affine_i])
                affine_i += 1

        pre_node_num = np.array(pre_node_num)

        if weight_init_std.lower() in ('relu', 'he'):
            weight_init_scale = np.sqrt(2.0 / pre_node_num, dtype=self.precision)
        if   weight_init_std.lower() in ('sigmoid', 'he'):
            weight_init_scale = np.sqrt(1.0 / pre_node_num, dtype=self.precision)


        #가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[-3] #입력과 필터의 채널 수는 동일해야 함함
        conv_i = 0
        affine_i = 0
        pooling_i = 0
        prev_node = 1
        for input_slice in input_dim: prev_node *= input_slice

        for layer in layer_list:
            idx  = conv_i + affine_i
            total_idx = idx + pooling_i

            if layer == 'conv':
                self.params[f"W{idx+1}"] = weight_init_scale[idx] * np.random.randn(
                    filter_shape_list[conv_i][FN], pre_channel_num, filter_shape_list[conv_i][FH], filter_shape_list[conv_i][FW], dtype=self.precision) #필터 크기 (FN, C, FH, FW)
                self.params[f"b{idx+1}"] = np.zeros(filter_shape_list[conv_i][FN], dtype=self.precision) #완전 연결 계층에서 출력층 노드에 한개씩 편향 적용 한 것처럼 필터 한개당 한개의 편향 적용
                
                prev_node = int(pre_node_num[conv_i])*filter_shape_list[conv_i][FN]
                pre_channel_num = filter_shape_list[conv_i][FN]

                conv_i += 1

            if layer == 'pooling':
                prev_node = int(pre_node_num[total_idx])
                pooling_i += 1

            if layer == 'affine':
                self.params[f"W{idx+1}"] = weight_init_scale[idx] * np.random.randn(prev_node ,hidden_layer_list[affine_i], dtype = self.precision)
                self.params[f'b{idx+1}'] = np.zeros(hidden_layer_list[affine_i], dtype = self.precision)
                prev_node = hidden_layer_list[affine_i]
                affine_i += 1

        #계층 생성
        conv_i = 0
        affine_i = 0
        self.layers = OrderedDict()
        self.layer_type = {'affine': Affine, 'conv': Convolution}
        self.activation_function = {'relu': Relu, 'sigmoid': Sigmoid}
        layer_with_weight = 0

        for layer in layer_list:

            if layer == 'softmax':
                self.last_layer = SoftmaxWithLoss(precision=self.precision)#Softmax층은 역전파시에만 활용하기 때문에 따로 제외
                break

            if layer in self.activation_function.keys():
                self.layers[f'activation_function{layer_with_weight}'] = self.activation_function[layer](precision = self.precision) 

            if layer == "pooling":
                pooling_size = pooling_list.pop()
                self.layers[f'pooling{layer_with_weight}'] = Pooling(pooling_size[0], pooling_size[1], pooling_size[2], precision=self.precision)

            if layer == 'dropout':
                self.layers[f'dropout{layer_with_weight}'] = Dropout(precision=self.precision)

            if layer in self.layer_type.keys():
                layer_with_weight += 1
                self.layers[layer+str(layer_with_weight)] = self.layer_type[layer](self.params[f'W{layer_with_weight}'], self.params[f'b{layer_with_weight}'],precision=self.precision) 

                if self.use_batchnorm:
                    output_node = output_node_num.popleft()
                    self.params[f'gamma{layer_with_weight}'] = np.ones(output_node, dtype = self.precision)
                    self.params[f'beta{layer_with_weight}'] = np.zeros(output_node, dtype = self.precision)
                    self.layers[f'batchnorm{layer_with_weight}'] = BatchNormalization(self.params[f'gamma{layer_with_weight}'], self.params[f'beta{layer_with_weight}'], precision=self.precision)

                if self.use_dropout:
                    self.layers[f'dropout{layer_with_weight}'] = Dropout(self.dropout_ration)

        self.layer_with_weight_num = layer_with_weight

        self.layers.pop(f'batchnorm{layer_with_weight}', None) #출력층에서는 배치정규화를 시행하지 않음 모델의 표현력 때문에 이 주석 무시
        self.params.pop(f'gamma{layer_with_weight}', None) #딕셔너리의 pop메서드는 pop(not in dictionray key, Argu1) -> return Argu1
        self.params.pop(f'beta{layer_with_weight}', None)

        self.layers.pop(f'dropout{layer_with_weight}', None)#출력층의 드롭아웃은 없음 있으면 출력층의 일부만 사용하게 됨됨
            
    def predict(self, x, train_flg = False):
        for key, layer in self.layers.items():
            if key in ('dropout', 'batchnorm'):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
            
        return x

    def loss(self, x, t, train_flg = True): #수정할것 태스트 동작후에
        y = self.predict(x, train_flg)

        weight_decay = 0
        idx = 1
        for layer in self.layers:
            if layer in self.layer_type.keys():
                W = self.params[f'W{idx}']
                idx += 1
                weight_decay += 0.5 *self.weight_decay_lambda * np.sum(W**2, dtype=self.precision)

        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, X, T, batch_size):
        start = 0
        Y = np.zeros((1, self.output_size))
        while start + batch_size < len(X):
            batch_x = X[start: start + batch_size]
            Y = np.concatenate((Y, self.predict(batch_x, train_flg=False)), 0)
            start += batch_size

        if start < X.size: Y = np.concatenate((Y, self.predict(batch_x, train_flg=False)), 0)

        if T.ndim != 1 : T = np.argmax(T, axis=1)

        Y = np.delete(Y, 0, 0)
        Y = np.argmax(Y, axis=1)
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return float(accuracy)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        idx = 1
        for layer in self.layers.keys():

            if 'conv' in layer or 'affine' in layer:
                grads[f'W{idx}'] = self.layers[layer].dW + self.weight_decay_lambda * self.params[f'W{idx}']
                grads[f'b{idx}'] = self.layers[layer].db

                if self.use_batchnorm and idx < self.layer_with_weight_num:
                    grads['gamma' + str(idx)] = self.layers['batchnorm' + str(idx)].dgamma
                    grads['beta' + str(idx)] = self.layers['batchnorm' + str(idx)].dbeta
                
                
                idx += 1
        return grads


    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        layer_num_with_weight = 0
        for layer_name, layer in self.layers.items():
            if 'conv' in layer_name or 'affine' in layer_name:
                layer.W = self.params['W' + str(layer_num_with_weight+1)]
                layer.b = self.params['b' + str(layer_num_with_weight+1)]

                layer_num_with_weight += 1
            
            if 'batchnorm' in layer_name.lower():
                layer.gamma = self.params[f'gamma{layer_num_with_weight}']
                layer.beta = self.params[f'beta{layer_num_with_weight}']

        print("Complete load Params data")