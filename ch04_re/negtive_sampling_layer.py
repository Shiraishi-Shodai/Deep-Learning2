import sys
sys.path.append("..")
from common.layers import Embedding
import numpy as np

class EmbeddingDot:
    def __init__(self, W) -> None:
        self.embed = Embedding(W)

        # 追加
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, index):
        target_W = self.embed.forward(index)
        out = np.sum(h * target_W, axis=1)

        self.cache = (h, target_W)

        return out
    
    def backward(self, dout):
        h, target_W = self.cache

        dout = dout.reshape(-1, 1)

        dtarget_W = dout * h # 列方向にdoutをhに掛ける

        self.embed.backward(dtarget_W)

        dh = dout * target_W

        return dh


