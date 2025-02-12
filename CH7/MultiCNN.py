import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
#import numpy as np
import cupy as np  
from collections import OrderedDict
from common.layers import *

FN, C,FH, FW, PAD, STRIDE = 0, 1, 2, 3, 4, 5


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
                 filter_shape_list = [[30, 5, 5, 0, 1]], hidden_layer_list = [[50],[50]], pooling_list = [[2, 2, 2], [2,2,2]], output_size = 10, activation = "ReLu", weight_init_std = 'ReLu', weight_decay_lambda=0,
                 use_dropout = False, dropout_ration = 0.5, use_batch_norm=False):
        layer_list = [layer.lower() for layer in layer_list ]
        try:
            assert layer_list.count('conv') == len(filter_shape_list), "Not match Convulution Layer`s numer and filter layer Number"
            assert layer_list.count('affine') == len(hidden_layer_list), "Not match Hidden Layer`s number and hidden layer list num"
            assert layer_list[-1] in ('softmax', 'identity'), "Output Layer`s activation function must be 'Softmax' or 'Identity function'"
            assert activation.lower() in ('relu', 'sigmoid'), "Not implement exclude relu and sigmoid activation function" 
            assert weight_init_std.lower() in ('relu', 'sigmoid', 'he', 'xavier'), "Not implement exclude relu and sigmoid activation function" 

            for filter_shape in filter_shape_list:
                assert len(filter_shape) == 5, "You have write 5 element"
        except:
            sys.exit()

        if not np.array(filter_shape_list).ndim == 2:
            filter_shape_list =[filter_shape_list] 
        if not np.array(hidden_layer_list).ndim == 2:
            hidden_layer_list = [[hidden_layer_list]]
        if not np.array(pooling_list) == 2:
            pooling_list = [pooling_list]


        conv_parm = []
        for filter_shape in filter_shape_list:
            conv_parm.append({'fliter_num': filter_shape[FN], 'filter_ch' : filter_shape[C],'fliter_h': filter_shape[FH], 'fliter_w': filter_shape[FW], 'pad': filter_shape[PAD], 'stride': filter_shape[STRIDE]})
        
        # 'he' 초깃값 활용 위해 앞 노드 개수 계산 conv계층은 필터 한개의 크기 c*fh*fw, affine계층은 앞 노드의 출력 노드 개수
        pre_node_num = [input_dim[0] * filter_shape_list[0][FH] * filter_shape_list[0][FW]] 
        fc = input_dim[-3]
        out_h = input_dim[-2]
        out_w = input_dim[-1]
        prev_node = 0
        conv_i = 0
        affine_i = 0


        for layer in layer_list:
            if layer == 'conv':
                out_h = (out_h + 2*filter_shape_list[conv_i][FH]) // filter_shape_list[conv_i][STRIDE] + 1  
                out_w = (out_w + 2*filter_shape_list[conv_i][FW]) // filter_shape_list[conv_i][STRIDE] + 1
                prev_node = fc * out_h * out_w 
                pre_node_num.append(prev_node) #cnn에서는 이미지 블록 한개의 크기  

                fc = filter_shape_list[conv_i][FN]
                conv_i += 1
            
            if layer == 'affine':
                pre_node_num.append(prev_node)
                prev_node = hidden_layer_list[affine_i]
                affine_i += 1

        if weight_init_std.lower() in ('relu', 'he'):
            weight_init_scale = np.sqrt(2.0 / pre_node_num)
        if   weight_init_std.lower() in ('sigmoid', 'he'):
            weight_init_scale = np.sqrt(1.0 / pre_node_num)


        #가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[-3] #입력과 필터의 채널 수는 동일해야 함함
        conv_i = 0
        affine_i = 0
        prev_node = 0
        hidden_layer_list.append(output_size)
        for input_slice in input_dim: prev_node += input_slice

        for layer in layer_list:
            idx  = conv_i + affine_i

            if layer == 'conv':
                self.params[f"W{idx+1}"] = weight_init_scale[idx] * np.random.randn(
                    filter_shape_list[conv_i][FN], pre_channel_num, filter_shape_list[conv_i][FH], filter_shape_list[conv_i][FW]) #필터 크기 (FN, C, FH, FW)
                self.params[f"b{idx+1}"] = np.zeros(filter_shape_list[conv_i][FN]) #완전 연결 계층에서 출력층 노드에 한개씩 편향 적용 한 것처럼 필터 한개당 한개의 편향 적용
                prev_node = pre_node_num[conv_i]*[filter_shape_list[FN]]
                pre_channel_num = filter_shape_list[conv_i][FN]
                conv_i += 1

            if layer == 'affine':
                self.params[f"W{idx+1}"] = weight_init_scale[idx * np.random.randn(pre_node_num ,hidden_layer_list[affine_i])]
                self.params[f'W{idx+1}'] = np.zeros(hidden_layer_list[affine_i])
                prev_node = hidden_layer_list[affine_i]
                affine_i += 1

        #계층 생성
        conv_i = 0
        affine_i = 0
        batch_norm_i = 0
        dropout_i = 0
        self.layers = OrderedDict()
        layer_type = {'affine': Affine, 'conv': Convolution, 'pooling': Pooling, 'relu': Relu,
                        'sigmoid': Sigmoid, 'dropout': Dropout, 'batchnorm': BatchNormalization}

            
            