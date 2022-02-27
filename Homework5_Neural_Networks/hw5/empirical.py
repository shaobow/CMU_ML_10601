import neuralnet as nn
import numpy as np
import matplotlib.pyplot as plt

exercise = '3.1c'  # '3.2'
if __name__ == '__main__':
    trainInput = 'data/small_train.csv'
    validInput = 'data/small_val.csv'
    initFlag = 1
    numEpoch = 100
    labelClass = {0, 1, 2, 3}  # hard-code the label class
    X_t, Y_t = nn.infile(trainInput, len(labelClass))
    X_v, Y_v = nn.infile(validInput, len(labelClass))

    for exercise in ['3.2']:
        if exercise == '3.1a':
            learnRate = 0.01
            hiddenUnit = np.array([5, 20, 50, 100, 200])
            y_train = np.zeros(hiddenUnit.shape)
            y_valid = np.zeros(hiddenUnit.shape)
            for i in range(hiddenUnit.size):  # , 50, 100, 200
                crossEntropy = np.zeros((2, numEpoch))
                Alpha, Beta = nn.init_weight(X_t.shape[1], hiddenUnit[i], len(labelClass), initFlag)
                StAlpha = np.zeros(Alpha.shape)
                StBeta = np.zeros(Beta.shape)  # init Adagrad intermediate value
                for NE in range(numEpoch):
                    Alpha, Beta, StAlpha, StBeta = nn.train_nn(Alpha, Beta, X_t, Y_t, learnRate, StAlpha, StBeta)
                    errorTrain, crossEntropy[[0], [NE]], labelTrain = nn.test_nn(Alpha, Beta, X_t, Y_t)
                    errorValid, crossEntropy[[1], [NE]], labelValid = nn.test_nn(Alpha, Beta, X_v, Y_v)
                y_train[i] = np.mean(crossEntropy, axis=1)[0]
                y_valid[i] = np.mean(crossEntropy, axis=1)[1]

            plt.figure(1, dpi=200)
            plt.plot(hiddenUnit, y_train, label='train')
            plt.plot(hiddenUnit, y_valid, label='valid')
            plt.legend()
            plt.xlabel('number of hidden units')
            plt.ylabel('average cross-entropy')
            plt.title('Avg. Train and Validation Cross-Entropy Loss')
            plt.show()
        elif exercise == '3.1c':
            lossSGD = np.loadtxt('val_loss_sgd_out.txt')

            hiddenUnit = 50
            learnRate = 0.01
            Alpha, Beta = nn.init_weight(X_t.shape[1], hiddenUnit, len(labelClass), initFlag)
            StAlpha = np.zeros(Alpha.shape)
            StBeta = np.zeros(Beta.shape)  # init Adagrad intermediate value
            lossAGD = np.zeros([numEpoch, ])
            for NE in range(numEpoch):
                Alpha, Beta, StAlpha, StBeta = nn.train_nn(Alpha, Beta, X_t, Y_t, learnRate, StAlpha, StBeta)
                errorValid, lossAGD[NE], labelValid = nn.test_nn(Alpha, Beta, X_v, Y_v)

            t = np.arange(numEpoch)
            plt.figure(2, dpi=200)
            plt.plot(t, lossSGD, label='SGD')
            plt.plot(t, lossAGD, label='Adagrad')
            plt.legend()
            plt.xlabel('number of epochs')
            plt.ylabel('average cross-entropy')
            plt.title('Avg. Validation Cross-Entropy Loss of SGD with and without Ada')
            plt.show()
        elif exercise == '3.2':
            hiddenUnit = 50
            X_t, Y_t = nn.infile(trainInput, len(labelClass))
            X_v, Y_v = nn.infile(validInput, len(labelClass))

            y_train = np.zeros([numEpoch, ])
            y_valid = np.zeros([numEpoch, ])
            t = np.arange(numEpoch)
            pltIdx = 3
            for learnRate in [0.1]:  # , 0.01, 0.001
                Alpha, Beta = nn.init_weight(X_t.shape[1], hiddenUnit, len(labelClass), initFlag)
                StAlpha = np.zeros(Alpha.shape)
                StBeta = np.zeros(Beta.shape)  # init Adagrad intermediate value
                for NE in range(numEpoch):
                    Alpha, Beta, StAlpha, StBeta = nn.train_nn(Alpha, Beta, X_t, Y_t, learnRate, StAlpha, StBeta)
                    errorTrain, y_train[NE], labelTrain = nn.test_nn(Alpha, Beta, X_t, Y_t)
                    errorValid, y_valid[NE], labelValid = nn.test_nn(Alpha, Beta, X_v, Y_v)
                plt.figure(pltIdx, dpi=200)
                plt.plot(t, y_train, label='train')
                plt.plot(t, y_valid, label='valid')
                plt.legend()
                plt.xlabel('number of epochs')
                plt.ylabel('average cross-entropy')
                plt.title(f'LR {learnRate}')
                plt.show()
                pltIdx += 1
