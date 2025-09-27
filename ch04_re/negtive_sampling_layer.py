import sys
sys.path.append("..")
from common.layers import Embedding
import numpy as np

class EmbeddingDot:
    def __init__(self, W) -> None:
        self.embed = Embedding(W)
    
    def forward(self, h, index):
        target_W = self.embed.forward(index)
        print(f"ターゲットW: {target_W}", end="\n\n")
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


