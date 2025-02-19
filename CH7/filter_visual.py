from matplotlib import pyplot as plt
import cupy as np
import pickle as pkl

def filter_show(filters, nx=4, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(show_num / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(show_num):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    
    plt.show()


with open('params.pkl', 'rb') as f: #학습된 가중치
#with open('init_params.pkl', 'rb') as f:  #초기상태
    params = pkl.load(f)


filter_show(params['W1'].get(), 16)
