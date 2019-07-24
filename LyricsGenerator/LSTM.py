import numpy as np


class LSTM:

    def __init__(self, x_size, y_size, learn_rate, seq_len):
        self.x = np.zeros(x_size + y_size)
        self.x_size = x_size + y_size

        self.cs = np.zeros(y_size)
        self.hs = np.zeros(y_size)
        self.y_size = y_size

        self.fW = np.random.random((y_size, x_size + y_size))
        self.iW = np.random.random((y_size, x_size + y_size))
        self.cW = np.random.random((y_size, x_size + y_size))
        self.oW = np.random.random((y_size, x_size + y_size))

        self.Gf = np.zeros_like(self.fW)
        self.Gi = np.zeros_like(self.iW)
        self.Gc = np.zeros_like(self.cW)
        self.Go = np.zeros_like(self.oW)

        self.learn_rate = learn_rate
        self.seq_len = seq_len

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tangent(self, x):
        return np.tanh(x)

    def dtangent(self, x):
        return 1 - np.tanh(x) ** 2

    def forwardProp(self):
        f = self.sigmoid(np.dot(self.fW, self.x))
        i = self.sigmoid(np.dot(self.iW, self.x))
        o = self.sigmoid(np.dot(self.oW, self.x))
        c = self.tangent(np.dot(self.cW, self.x))
        self.cs = f * self.cs + i * c
        self.hs = o * self.tangent(self.cs)
        return self.cs, self.hs, f, i, c, o

    def backProp(self, prev_cs, cs, f, i, c, o, err_hs, err_cs):
        err_hs = np.clip(err_hs, -6, 6)
        p = np.clip(err_hs * o * self.dtangent(cs) + err_cs, -6, 6)
        q = err_hs * self.tangent(cs)
        w = q * o * (1 - o)
        x = p * c * i * (1 - i)
        y = p * i * (1 - c ** 2)
        z = p * prev_cs * f * (1 - f)
        oU = np.dot(np.atleast_2d(w).T, np.atleast_2d(self.x))
        iU = np.dot(np.atleast_2d(x).T, np.atleast_2d(self.x))
        cU = np.dot(np.atleast_2d(y).T, np.atleast_2d(self.x))
        fU = np.dot(np.atleast_2d(z).T, np.atleast_2d(self.x))
        err_prev_cs = p * f
        err_prev_hs = (np.dot(w, self.oW) +
                       np.dot(x, self.iW) +
                       np.dot(y, self.cW) +
                       np.dot(z, self.fW))[0:self.y_size]

        return fU, oU, iU, cU, err_prev_cs, err_prev_hs

    def update(self, fU, iU, cU, oU):
        self.Gf = 0.9 * self.Gf + 0.1 * fU ** 2
        self.Gi = 0.9 * self.Gi + 0.1 * iU ** 2
        self.Gc = 0.9 * self.Gc + 0.1 * cU ** 2
        self.Go = 0.9 * self.Go + 0.1 * oU ** 2

        self.fW -= (self.learn_rate * fU) / np.sqrt(self.Gf + 1e-8)
        self.iW -= (self.learn_rate * iU) / np.sqrt(self.Gi + 1e-8)
        self.cW -= (self.learn_rate * cU) / np.sqrt(self.Gc + 1e-8)
        self.oW -= (self.learn_rate * oU) / np.sqrt(self.Go + 1e-8)

    def print(self):
        print("LSTM:")
        print("x_size->", end="")
        print(self.x_size)
        print("y_size->", end="")
        print(self.y_size)
        print("learn_rate->", end="")
        print(self.learn_rate)
        print("seq_len->", end="")
        print(self.seq_len)



