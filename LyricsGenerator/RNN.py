import numpy as np
import LSTM as ls


class RNN:

    def __init__(self, x_size, y_size, learn_rate, seq_len, expected_out):
        self.x = np.zeros(x_size)
        self.x_size = x_size
        self.y = np.zeros(y_size)
        self.y_size = y_size

        # information floating out of LSTM cell(hiddenState, cellState, outputState)
        self.hsArray = np.zeros((seq_len + 1, y_size))
        self.csArray = np.zeros((seq_len + 1, y_size))
        self.osArray = np.zeros((seq_len + 1, y_size))

        # information about gates(forgot, input, output, cellState) inside LSTM cell
        self.fArray = np.zeros((seq_len + 1, y_size))
        self.iArray = np.zeros((seq_len + 1, y_size))
        self.oArray = np.zeros((seq_len + 1, y_size))
        self.cArray = np.zeros((seq_len + 1, y_size))

        # weight matrix
        self.w = np.random.random((y_size, y_size))
        self.G = np.zeros_like(self.w)

        # common parameters
        self.learn_rate = learn_rate
        self.seq_len = seq_len
        self.expected_out = np.vstack((np.zeros((1, x_size)), expected_out))

        self.LSTM = ls.LSTM(x_size, y_size, learn_rate, seq_len)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forwardProp(self):
        for j in range(1, self.seq_len + 1):
            self.x = self.expected_out[j - 1]
            self.LSTM.x = np.hstack((self.hsArray[j - 1], self.x))
            cs, hs, f, i, c, o = self.LSTM.forwardProp()
            self.fArray[j] = f
            self.iArray[j] = i
            self.oArray[j] = o
            self.cArray[j] = c

            self.csArray[j] = cs
            self.hsArray[j] = hs
            self.osArray[j] = self.sigmoid(np.dot(self.w, hs))

        return self.osArray

    def backProp(self):
        totalError = 0
        tfU = np.random.random((self.y_size, self.y_size + self.x_size))
        tiU = np.random.random((self.y_size, self.y_size + self.x_size))
        toU = np.random.random((self.y_size, self.y_size + self.x_size))
        tcU = np.random.random((self.y_size, self.y_size + self.x_size))
        twU = np.random.random((self.y_size, self.y_size))

        err_hs = np.zeros(self.y_size)
        err_cs = np.zeros(self.y_size)

        for i in range(self.seq_len, 0, -1):
            error = self.osArray[i] - self.expected_out[i]
            p = error * self.osArray[i] * (1 - self.osArray[i])
            wU = np.dot(np.atleast_2d(p).T , np.atleast_2d(self.hsArray[i]))
            err_hs += np.dot(p, self.w)   # here should be a dot product
            self.LSTM.x = np.hstack((self.hsArray[i - 1], self.expected_out[i - 1]))
            fU, oU, iU, cU, err_cs, err_hs = \
                self.LSTM.backProp(self.csArray[i-1], self.csArray[i],
                                   self.fArray[i], self.iArray[i], self.cArray[i],
                                   self.oArray[i], err_hs, err_cs)
            tfU += fU
            toU += oU
            tiU += iU
            tcU += cU
            twU += wU
            totalError += np.sum(error ** 2)

        self.update(tfU/self.seq_len, toU/self.seq_len, tcU/self.seq_len, tiU/self.seq_len, twU/self.seq_len)
        return totalError/self.seq_len

    def update(self,tfU, toU, tcU, tiU, twU):
        self.G = 0.9 * self.G + 0.1 * (twU ** 2)
        self.w -= (self.learn_rate * twU)/np.sqrt(self.G + 1e-8)
        self.LSTM.update(tfU, tiU, tcU, toU)

    def get_val(self):
        return self.w, self.LSTM.fW, self.LSTM.iW, self.LSTM.cW, self.LSTM.oW

    def set_val(self, w, fW, iW, cW, oW):
        self.w = w
        self.LSTM.fW = fW
        self.LSTM.iW = iW
        self.LSTM.cW = cW
        self.LSTM.oW = oW

    def print(self):
        print("RNN:")
        print("x_size->", end="")
        print(self.x_size)
        print("y_size->", end="")
        print(self.y_size)
        print("learn_rate->", end="")
        print(self.learn_rate)
        print("seq_len->", end="")
        print(self.seq_len)


def main():
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    rnn = RNN(3, 3, 0.01, 4, a)
    print(rnn.forwardProp())
    print(rnn.backProp())


if __name__ == '__main__':
    main()