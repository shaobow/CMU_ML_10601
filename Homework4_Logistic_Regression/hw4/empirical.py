import matplotlib.pyplot as plt
import numpy as np
import feature
import lr

if __name__ == '__main__':
    trainInput = 'largedata/train_data.tsv'
    validInput = 'largedata/valid_data.tsv'
    dictInput = 'dict.txt'
    fTrainOutput = 'largeoutput/formatted_train.tsv'
    fValidOutput = 'largeoutput/formatted_valid.tsv'

    featDictInput = 'word2vec.txt'
    fTrainInput = fTrainOutput
    fTestInput = fValidOutput

    featFlag = 1
    numEpoch = 5000

    '''feature engineering'''
    # (trainLabel, trainText) = feature.review_input(trainInput)
    # (validLabel, validText) = feature.review_input(validInput)
    # if featFlag == 1:
    #     dictVocab = feature.dict_input(dictInput)
    #     fmtTrain = feature.model1(dictVocab, trainText, trainLabel)
    #     fmtValid = feature.model1(dictVocab, validText, validLabel)
    #     # feature vector for model 1 bag-of-words in stack of rows
    #     feature.feat_output(fTrainOutput, fmtTrain, '%s')
    #     feature.feat_output(fValidOutput, fmtValid, '%s')
    # elif featFlag == 2:
    #     (featVocab, vocabWeight) = feature.featdict_input(featDictInput)
    #     fmtTrain = feature.model2(featVocab, vocabWeight, trainText, trainLabel)
    #     fmtValid = feature.model2(featVocab, vocabWeight, validText, validLabel)
    #     # feature vector for model 2 word2vec in stack of rows
    #     feature.feat_output(fTrainOutput, fmtTrain, '%1.6f')
    #     feature.feat_output(fValidOutput, fmtValid, '%1.6f')

    '''logistic regression'''
    (trainLabel, trainFeat) = lr.example_input(fTrainOutput)
    (validLabel, validFeat) = lr.example_input(fValidOutput)

    def sigmoid(t):
        sig = 1 / (1 + np.exp(-t))
        return sig

    def cost_func(example_t, example_v, label_t, label_v, epoch):
        theta = np.zeros((example_t.shape[1] + 1, 1))
        alpha = 0.01
        num_t = label_t.size
        num_v = label_v.size
        cost_all_t = np.empty(0,)
        cost_all_v = np.empty(0, )
        y_all_t = label_t.reshape((1, num_t))
        y_all_v = label_v.reshape((1, num_v))
        x_all_t = np.concatenate((np.ones((1, num_t)), example_t.T), axis=0)
        x_all_v = np.concatenate((np.ones((1, num_v)), example_v.T), axis=0)
        for k in range(epoch * num_t):
            i = k % num_t
            x_i = x_all_t[:, [i]]
            # np.concatenate(([[1]], example[[i], :].T), axis=0)
            y_i = label_t[[i], [0]]
            phi_i = sigmoid(np.dot(theta.T, x_i))
            theta += alpha / num_t * (y_i - phi_i) * x_i
            if i == num_t-1:
                phi_t = sigmoid(np.dot(theta.T, x_all_t))
                phi_v = sigmoid(np.dot(theta.T, x_all_v))
                cost_t = -np.sum(y_all_t*np.log(phi_t)+(1-y_all_t)*np.log(1-phi_t), axis=1) / num_t
                cost_all_t = np.append(cost_all_t, cost_t)
                cost_v = -np.sum(y_all_v*np.log(phi_v)+(1-y_all_v)*np.log(1-phi_v), axis=1) / num_v
                cost_all_v = np.append(cost_all_v, cost_v)
        return cost_all_t, cost_all_v

    def compare_alpha(example_t, label_t, epoch, alpha):
        theta = np.zeros((example_t.shape[1] + 1, 1))
        num_t = label_t.size
        cost_all_t = np.empty(0,)
        y_all_t = label_t.reshape((1, num_t))
        x_all_t = np.concatenate((np.ones((1, num_t)), example_t.T), axis=0)
        for k in range(epoch * num_t):
            i = k % num_t
            x_i = x_all_t[:, [i]]
            # np.concatenate(([[1]], example[[i], :].T), axis=0)
            y_i = label_t[[i], [0]]
            phi_i = sigmoid(np.dot(theta.T, x_i))
            theta += alpha / num_t * (y_i - phi_i) * x_i
            if i == num_t-1:
                phi_t = sigmoid(np.dot(theta.T, x_all_t))
                cost_t = -np.sum(y_all_t*np.log(phi_t)+(1-y_all_t)*np.log(1-phi_t), axis=1) / num_t
                cost_all_t = np.append(cost_all_t, cost_t)
        return cost_all_t

    for learnRate in {0.001, 0.01, 0.1}:
        cost_train = compare_alpha(trainFeat, trainLabel, numEpoch, learnRate)
        x = np.arange(numEpoch)
        Y_train = [cost_train[x_i] for x_i in x]
        plt.figure(1, dpi=300)
        plt.plot(x, Y_train, label=f'Train_ANLL_alpha={learnRate}')
    plt.legend()
    plt.xlabel("# epochs")
    plt.ylabel("Average negative log likelihood (ANLL)")
    plt.title("Average negative log likelihood against number of epochs")
    plt.show()
