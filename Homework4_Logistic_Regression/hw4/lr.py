import sys
import numpy as np


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def example_input(filedir):
    infile = np.loadtxt(filedir, dtype=float, delimiter='\t')
    label = infile[:, [0]]
    feat = infile[:, 1:]
    return label, feat


def train_theta(example, label, epoch):
    theta = np.zeros((example.shape[1] + 1, 1))
    alpha = 0.01
    num = label.size
    for k in range(epoch * num):
        i = k % num
        x_i = np.concatenate(([[1]], example[[i], :].T), axis=0)
        y_i = label[[i], [0]]
        phi_i = sigmoid(np.dot(theta.T, x_i))
        theta += alpha / num * (y_i - phi_i) * x_i
    return theta


def test_theta(example, label, theta):
    num = label.size
    y_h = np.empty((num, 1))
    for i in range(num):
        x_i = np.concatenate(([[1]], example[[i], :].T), axis=0)
        y_h[[i]] = sigmoid(np.dot(theta.T, x_i))
    y_h = np.asarray(y_h >= 0.5).astype(int)
    error = np.sum(y_h != label) / num
    return error, y_h


def label_output(filedir, label, fmt):
    np.savetxt(filedir, label, fmt=fmt, newline='\n')


def metrics_output(filedir, text):
    with open(filedir, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    args = sys.argv
    assert (len(args) == 9)
    fTrainInput = args[1]
    fValidInput = args[2]
    fTestInput = args[3]
    dictInput = args[4]
    trainOutput = args[5]
    testOutput = args[6]
    metricsOutput = args[7]
    numEpoch = int(args[8])

    # fTrainInput = 'smalloutput/formatted_train.tsv'
    # fTestInput = 'smalloutput/formatted_test.tsv'
    # trainOutput = 'smalloutput/train_out.labels'
    # testOutput = 'smalloutput/test_out.labels'
    # metricsOutput = 'smalloutput/metrics_out.txt'
    # numEpoch = 500

    (trainLabel, trainFeat) = example_input(fTrainInput)
    (testLabel, testFeat) = example_input(fTestInput)
    param = train_theta(trainFeat, trainLabel, numEpoch)
    (trainErr, trainPred) = test_theta(trainFeat, trainLabel, param)
    (testErr, testPred) = test_theta(testFeat, testLabel, param)

    generalMetrics = (f"error(train): {trainErr}"+"\n"+f"error(test): {testErr}")

    # label_output(trainOutput, trainPred, '%s')
    # label_output(testOutput, testPred, '%s')
    metrics_output(metricsOutput, generalMetrics)

'''
python lr.py largeoutput/model1_formatted_train.tsv largeoutput/model1_formatted_valid.tsv largeoutput/model1_formatted_test.tsv dict.txt largeoutput/train_out.labels largeoutput/test_out.labels largeoutput/HW_model1_metrics_out.txt 5000
'''