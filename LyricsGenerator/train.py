import word_embeddings as glove
import RNN
import numpy as np


class ExpectedOut:

    def __init__(self, file_name):
        cursorPos = int(open('rnn_char/cursorPos.txt').read().split()[0])
        self.cursorPos = cursorPos
        self.file_name = file_name
        self.total_len = 57650

    def get_batch(self):
        if self.total_len == 0:
            self.total_len = 57650
            self.cursorPos = 0
        with open(self.file_name, 'r') as data:
            data.seek(self.cursorPos)
            line = data.readline()
            self.cursorPos += len(line)
            self.total_len -= 50
            vecs = list(map(glove.word2vec, line.split(" ")))
            return np.array([vec for vec in vecs])

    def initiate_again(self):
        with open('rnn_char/cursorPos.txt', 'w') as f:
            f.write(str(0))
        np.savetxt('rnn_char/weight.txt', np.random.rand(100, 100), fmt=['%1.5e' for i in range(100)])
        np.savetxt('rnn_char/forgetW.txt', np.random.rand(100, 200), fmt=['%1.5e' for i in range(200)])
        np.savetxt('rnn_char/inputW.txt', np.random.rand(100, 200), fmt=['%1.5e' for i in range(200)])
        np.savetxt('rnn_char/cellW.txt', np.random.rand(100, 200), fmt=['%1.5e' for i in range(200)])
        np.savetxt('rnn_char/outputW.txt', np.random.rand(100, 200), fmt=['%1.5e' for i in range(200)])


def main():

    EO = ExpectedOut('data/formattedSongs.txt')
    EO.initiate_again()
    eo = EO.get_batch()
    rnn = RNN.RNN(100, 100, 0.3, len(eo), eo)
    w = np.loadtxt('rnn_char/weight.txt', dtype=float)
    fW = np.loadtxt('rnn_char/forgetW.txt', dtype=float)
    iW = np.loadtxt('rnn_char/inputW.txt', dtype=float)
    cW = np.loadtxt('rnn_char/cellW.txt', dtype=float)
    oW = np.loadtxt('rnn_char/outputW.txt', dtype=float)
    rnn.set_val(w, fW, iW, cW, oW)
    rnn.forwardProp()
    rnn.backProp()
    # batch_size = min(50, EO.total_len)
    for i in range(57650):
        print(i)
        eo = EO.get_batch()
        rnn = RNN.RNN(100, 100, 0.3, len(eo), eo)
        w, fW, iW, cW, oW = rnn.get_val()
        rnn.set_val(w, fW, iW, cW, oW)
        rnn.forwardProp()
        rnn.backProp()

    print(rnn.get_val())
    w, fW, iW, cW, oW = rnn.get_val()
    with open('rnn_char/cursorPos.txt', 'w') as f:
        f.write(str(EO.cursorPos))
    np.savetxt('rnn_char/weight.txt', w, fmt=['%1.5e' for i in range(100)])
    np.savetxt('rnn_char/forgetW.txt', fW, fmt=['%1.5e' for i in range(200)])
    np.savetxt('rnn_char/inputW.txt', iW, fmt=['%1.5e' for i in range(200)])
    np.savetxt('rnn_char/cellW.txt', cW, fmt=['%1.5e' for i in range(200)])
    np.savetxt('rnn_char/outputW.txt', oW, fmt=['%1.5e' for i in range(200)])


def test():
    with open('rnn_char/cursorPos.txt', 'r') as f:
        print(f.read())
    w = np.loadtxt('rnn_char/weight.txt', dtype=float)
    fW = np.loadtxt('rnn_char/forgetW.txt', dtype=float)
    iW = np.loadtxt('rnn_char/inputW.txt', dtype=float)
    cW = np.loadtxt('rnn_char/cellW.txt', dtype=float)
    oW = np.loadtxt('rnn_char/outputW.txt', dtype=float)
    print(w[3, 3])
    print(fW[3, 3])
    print(iW[3, 3])
    print(cW[3, 3])
    print(oW[3, 3])
    print(cW[3, 3])
    print(oW[3, 3])


if __name__ == '__main__':
    main()
