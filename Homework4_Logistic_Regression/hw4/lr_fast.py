
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
    theta = np.zeros((example.shape[1] + 1, 1), dtype=float)
    alpha = 0.01
    num = label.size
    x_all = np.concatenate((np.ones((1, num)), example.T), axis=0, dtype=float)
    y_all = label.T
    for k in range(epoch * num):
        i = k % num
        x_i = x_all[:, [i]]
        y_i = y_all[:, [i]]
        phi_i = sigmoid(np.dot(theta.T, x_i))
        theta += alpha * (y_i - phi_i) * x_i
    theta = theta/num
    return theta


def test_theta(example, label, theta):
    num = label.size
    x_all = np.concatenate((np.ones((1, num)), example.T), axis=0, dtype=float)
    y_h = sigmoid(np.dot(theta.T, x_all))
    y_h = np.asarray(y_h >= 0.5).T
    error = np.sum(y_h != label) / num
    return error


def label_output(filedir, label, fmt):
    np.savetxt(filedir, label, fmt=fmt, newline='\n')


def metrics_output(filedir, text):
    with open(filedir, 'w') as f:
        f.write(text)


if __name__ == '__main__':

    for i in {1, 2}:
        fTrainInput = f'largeoutput/model{i}_formatted_train.tsv'
        fTestInput = f'largeoutput/model{i}_formatted_test.tsv'
        metricsOutput = f'largeoutput/HW_model{i}_metrics_out.txt'
        numEpoch = 5000

        (trainLabel, trainFeat) = example_input(fTrainInput)
        (testLabel, testFeat) = example_input(fTestInput)

        param = train_theta(trainFeat, trainLabel, numEpoch)
        trainErr = test_theta(trainFeat, trainLabel, param)
        testErr = test_theta(testFeat, testLabel, param)

        print(trainErr)
        print(testErr)

        generalMetrics = (f"error(train): {trainErr}"+"\n"+f"error(test): {testErr}")
        metrics_output(metricsOutput, generalMetrics)
