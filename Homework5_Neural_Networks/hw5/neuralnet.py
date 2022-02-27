import sys
import numpy as np


def encode_oh(array_in, num_class):  # transform label into one-hot encoding
    array_len = array_in.shape[0]
    array_out = np.zeros((array_len, num_class))
    array_out[np.arange(array_len), array_in.flatten()] = 1  # label class starts from 0
    return array_out.astype(int)


def decode_oh(array_in):
    array_out = np.argwhere(array_in == 1)[:, [1]]
    return array_out


def infile(filedir, num_class):
    infile_array = np.loadtxt(filedir, delimiter=',')
    label = infile_array[:, [0]].astype(int)
    label = encode_oh(label, num_class)
    pixel = infile_array[:, 1:]
    return pixel, label


def add_zeros(array_in):  # add zero at the beginning of each row
    zeros = np.zeros((array_in.shape[0], 1))
    array_out = np.concatenate((zeros, array_in), axis=1)
    return array_out


def init_weight(num_pixel, num_hid, num_class, init_flag):
    if init_flag == 1:
        theta_1 = np.random.uniform(-0.1, 0.1, (num_hid, num_pixel))
        theta_2 = np.random.uniform(-0.1, 0.1, (num_class, num_hid))
        theta_1 = add_zeros(theta_1)
        theta_2 = add_zeros(theta_2)  # return bias weight
        return theta_1, theta_2
    elif init_flag == 2:
        theta_1 = np.zeros((num_hid, num_pixel))
        theta_2 = np.zeros((num_class, num_hid))
        theta_1 = add_zeros(theta_1)
        theta_2 = add_zeros(theta_2)
        return theta_1, theta_2


def sigmoid(array_in):
    array_out = 1 / (1 + np.exp(-array_in))
    return array_out


def softmax(array_in):
    array_out = np.exp(array_in) / np.sum(np.exp(array_in))
    return array_out


def ce_loss(array_in, true_label, num):
    loss_out = np.dot(-true_label.T, np.log(array_in))/num
    return loss_out


def descent_ada(st, eta, theta, grad):
    epsilon = 1e-5
    st += np.power(grad, 2)
    theta -= eta/np.sqrt(st+epsilon)*grad
    return st, theta


def train_nn(theta_1, theta_2, x_all, y_all, eta, st1, st2):  # , ce_flag
    num = x_all.shape[0]
    for i in range(num):
        x_i = x_all[[i], :]
        y_i = y_all[[i], :]  # ith row
        train = NeuralNetsOneHid(x_i, y_i, theta_1, theta_2, num)
        grad_1, grad_2 = train.back_prop()
        '''check grad calculation'''
        # if np.sum(train.back_prop()[0]-train.finite_diff()[0]) >= 1e-5 or np.sum(train.back_prop()[1]-train.finite_diff()[1]) >= 1e-5:
        #     print(train.back_prop()[0])
        #     print(train.finite_diff()[0])
        st1, theta_1 = descent_ada(st1, eta, theta_1, grad_1)
        st2, theta_2 = descent_ada(st2, eta, theta_2, grad_2)
        if i < 5:
            print("Alpha:")
            print(theta_1)
            print("Beta:")
            print(theta_2)
            print(train.z)
    return theta_1, theta_2, st1, st2


def test_nn(theta_1, theta_2, x_all, y_all):
    num = x_all.shape[0]
    y_hat_all = np.zeros((num, 1)).astype(int)
    y_all_noh = decode_oh(y_all)  # none one-hot true label
    ce = np.zeros((num, 1))
    for i in range(num):
        x_i = x_all[[i], :]
        y_i = y_all[[i], :]  # ith row
        test = NeuralNetsOneHid(x_i, y_i, theta_1, theta_2, num)
        y_hat_all[[i], :] = test.label_predict().astype(int)  # none one-hot prediction
        ce[[i], :] = test.loss
    err = np.count_nonzero(y_hat_all != y_all_noh) / num
    ce_mean = np.sum(ce)
    return err, ce_mean, y_hat_all


class NeuralNetsOneHid(object):
    def __init__(self, x, y, alpha, beta, num):  # input weights contain bias
        self.x, self.y = x.T, y.T  # x,y are column vectors
        self.x_bias = np.concatenate(([[1]], self.x), axis=0)
        self.alpha = alpha[:, 1:]  # exclude bias
        self.beta = beta[:, 1:]
        self.alpha_bias, self.beta_bias = alpha, beta
        # forward prop
        self.a = np.dot(self.alpha_bias, self.x_bias)
        self.z = sigmoid(self.a)
        self.z_bias = np.concatenate(([[1]], self.z), axis=0)
        self.b = np.dot(self.beta_bias, self.z_bias)
        self.y_hat = softmax(self.b)
        self.num = num
        self.loss = ce_loss(self.y_hat, self.y, self.num)

    def back_prop(self):
        dldb = (self.y_hat * np.sum(self.y) - self.y).T  # dldb is row vector
        dldbeta = np.dot(dldb.T, self.z_bias.T)
        dldz = np.dot(dldb, self.beta)  # use beta without bias
        dlda = dldz * (self.z * (1 - self.z)).T
        dldalpha = np.dot(dlda.T, self.x_bias.T)
        return dldalpha, dldbeta

    def label_predict(self):
        return np.argmax(self.y_hat)

    '''---------------------finite_diff---------------------------'''
    # def finite_diff(self):
    #     epsilon = 1e-5
    #     x = self.x.T
    #     y = self.y.T
    #     alpha = self.alpha_bias
    #     beta = self.beta_bias
    #
    #     grad_alpha = np.zeros(alpha.shape)
    #     for i in range(alpha.shape[0]):
    #         for j in range(alpha.shape[1]):
    #             d = np.zeros(alpha.shape)
    #             d[[i], [j]] = 1
    #             v_alpha = NeuralNetsOneHid(x, y, alpha + epsilon * d, beta).loss
    #             v_alpha -= NeuralNetsOneHid(x, y, alpha - epsilon * d, beta).loss
    #             v_alpha /= 2 * epsilon
    #             grad_alpha[[i], [j]] = v_alpha
    #
    #     grad_beta = np.zeros(beta.shape)
    #     for i in range(beta.shape[0]):
    #         for j in range(beta.shape[1]):
    #             d = np.zeros(beta.shape)
    #             d[[i], [j]] = 1
    #             v_beta = NeuralNetsOneHid(x, y, alpha, beta + epsilon * d).loss
    #             v_beta -= NeuralNetsOneHid(x, y, alpha, beta - epsilon * d).loss
    #             v_beta /= 2 * epsilon
    #             grad_beta[[i], [j]] = v_beta
    #     return grad_alpha, grad_beta
    '''------------------------------------------------------------'''


if __name__ == '__main__':
    args = sys.argv
    assert (len(args) == 10)
    trainInput = args[1]
    validInput = args[2]
    trainOutput = args[3]
    validOutput = args[4]
    metricsOutput = args[5]
    numEpoch = int(args[6])
    hiddenUnit = int(args[7])
    initFlag = int(args[8])
    learnRate = float(args[9])

    # '''-------------------------debug------------------------------'''
    # trainInput = 'data/small_train.csv'
    # validInput = 'data/small_val.csv'
    # trainOutput = 'smallTrain_out.labels'
    # validOutput = 'smallValidation_out.labels'
    # metricsOutput = 'smallMetrics_out.txt'
    # numEpoch = 2
    # hiddenUnit = 4
    # initFlag = 2
    # learnRate = 0.1
    # '''-------------------------------------------------------------'''

    labelClass = {0, 1, 2, 3}  # hard-code the label class

    crossEntropy = np.zeros((2, numEpoch))  # init cross entropy result matrix

    X_t, Y_t = infile(trainInput, len(labelClass))  # training data input
    X_v, Y_v = infile(validInput, len(labelClass))  # validation data input

    Alpha, Beta = init_weight(X_t.shape[1], hiddenUnit, len(labelClass), initFlag)
    StAlpha = np.zeros(Alpha.shape)
    StBeta = np.zeros(Beta.shape)  # init Adagrad intermediate value

    for NE in range(numEpoch):
        # train over one epoch
        Alpha, Beta, StAlpha, StBeta = train_nn(Alpha, Beta, X_t, Y_t, learnRate, StAlpha, StBeta)
        # test over one epoch
        errorTrain, crossEntropy[[0], [NE]], labelTrain = test_nn(Alpha, Beta, X_t, Y_t)
        errorValid, crossEntropy[[1], [NE]], labelValid = test_nn(Alpha, Beta, X_v, Y_v)

    '''-------------------------------------------------Output-----------------------------------------------------------------------'''
    np.savetxt(trainOutput, labelTrain, newline='\n', fmt='%s')
    np.savetxt(validOutput, labelValid, newline='\n', fmt='%s')

    with open(metricsOutput, 'w') as f:
        for NE in range(numEpoch):
            f.write(f"epoch={NE+1} corssentropy(train): {float(crossEntropy[[0], [NE]])}\n")
            f.write(f"epoch={NE+1} corssentropy(validation): {float(crossEntropy[[1], [NE]])}\n")
        f.write(f"error(train): {errorTrain}\n")
        f.write(f"error(validation): {errorValid}\n")

    ''' 
    python neuralnet.py smallTrain.csv smallValidation.csv smallTrain_out.labels smallValidation_out.labels smallMetrics_out.txt 2 4 2 0.1
    '''
