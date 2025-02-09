import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from common.util import im2col

x1 = np.random.randn(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=2)
print(col1.shape)