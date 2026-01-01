import sys
sys.path.append("..")
import numpy as np
from ch05_re.time_layers import *

class RNNRegressor:
    def __init__(self, all_data_size, vec_size=100, hidden1_size=100, hidden2_size=1):
        V, D, H1, H2 = all_data_size, vec_size, hidden1_size, hidden2_size
        rn = np.random.randn

        affine1_W = (rn(V, D) / 100).astype("f")
        affine1_b = np.zeros(H).astype("f")
        rnn_Wx = (rn(D, H1) / np.sqrt(D)).astype("f")
        rnn_Wh = (rn(H1, H1) / np.sqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")
        affine2_W = (rn(H1, H2) / np.sqrt(H1)).astype("f")
        affine2_b = np.zeros(H2).astype("f")

        self.layers = [
            TimeAffine(affine1_W, affine1_b),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b),
            TimeAffine(affine2_W, affine2_b)
        ]

        self.loss_layer = TimeReLUWithLoss()
        self.rnn_layer = [
            self.layers[1]
        ]

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs):
        out = xs
        for layer in self.layers:
            out = layer.forward(out)
        
        return out
        
    def forward(self, xs, ts):
        out = self.predict(xs)
        loss = self.loss_layer.forward(out, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout
    
    def reset_state(self):
    """隠れ状態hをNoneにする
    """
        for rnn_layer in self.lstm_layers:
            rnn_layer.reset_state()