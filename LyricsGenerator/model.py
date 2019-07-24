import RNN
import numpy as np
import word_embeddings as glove


class model:
    def __init__(self, seed, length):
        self.rnn = RNN.RNN(100, 100, 0.3, length, [np.random.rand(100) for i in range(length)])
        w = np.loadtxt('rnn_char/weight.txt', dtype=float)
        fW = np.loadtxt('rnn_char/forgetW.txt', dtype=float)
        iW = np.loadtxt('rnn_char/inputW.txt', dtype=float)
        cW = np.loadtxt('rnn_char/cellW.txt', dtype=float)
        oW = np.loadtxt('rnn_char/outputW.txt', dtype=float)
        self.rnn.set_val(w, fW, iW, cW, oW)
        self.seed = seed
        self.length = length
        self.out = None

    def generate(self):
        self.rnn.osArray[0] = glove.word2vec(self.seed)
        for j in range(1, self.length+1):
            self.rnn.x = self.rnn.osArray[j - 1]
            self.rnn.LSTM.x = np.hstack((self.rnn.hsArray[j - 1], self.rnn.x))
            cs, hs, f, i, c, o = self.rnn.LSTM.forwardProp()
            self.rnn.fArray[j] = f
            self.rnn.iArray[j] = i
            self.rnn.oArray[j] = o
            self.rnn.cArray[j] = c

            self.rnn.csArray[j] = cs
            self.rnn.hsArray[j] = hs
            self.rnn.osArray[j] = self.rnn.sigmoid(np.dot(self.rnn.w, hs))
        self.out = []
        for n, os in enumerate(self.rnn.osArray):
            print(n)
            self.out.append(glove.vec2word(os))
        return self.out


def main():
    dummy_model = model("look", 30)
    print(dummy_model.generate())


if __name__ == '__main__':
    main()