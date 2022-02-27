import sys
import numpy as np


def review_input(filedir):
    label, text = np.genfromtxt(filedir, dtype='str', comments=None, delimiter='\t', usecols=(0, 1), unpack=True)
    return label.astype(int), text
    # return 1-d array


def dict_input(filedir):
    vocab = np.genfromtxt(filedir, dtype='str', comments=None, delimiter=' ', usecols=0, unpack=True)
    return vocab


def featdict_input(filedir):
    infile = np.genfromtxt(filedir, dtype='str', comments=None, delimiter='\t', unpack=True)
    vocab = infile[0, :]
    weight = infile[1:, :]
    return vocab, weight.astype(float)


def model1(vocab, text, label):
    vec = np.empty((0, vocab.size))  # feature vector for model 1 bag-of-words in stack of rows
    for i in range(text.size):
        sep_i = np.asarray(text[i].split(' '))  # separate text in to array with delimiter
        vec_i = np.isin(vocab, sep_i, assume_unique=True).reshape((1, vocab.size))
        vec = np.append(vec, vec_i, axis=0)
    vec = np.append(label.reshape((label.size, 1)), vec.astype(int), axis=1)
    return vec


def model2(vocab, weight, text, label):
    vec = np.empty((0, weight.shape[0]))
    for i in range(text.size):
        repeat = np.empty((1, vocab.size))
        sep_i = np.asarray(text[i].split(' '))
        for j in range(vocab.size):
            repeat[:, [j]] = np.count_nonzero(sep_i == vocab[j])
        vec_i = np.dot(repeat, weight.T) / np.sum(repeat)
        vec = np.append(vec, vec_i, axis=0)
    vec = np.append(label.reshape((label.size, 1)), vec, axis=1)
    return vec


def feat_output(filedir, array, fmt):
    np.savetxt(filedir, array, delimiter='\t', fmt=fmt, newline='\n')


if __name__ == '__main__':
    args = sys.argv
    assert (len(args) == 10)
    trainInput = args[1]
    validInput = args[2]
    testInput = args[3]
    dictInput = args[4]
    fTrainOutput = args[5]
    fValidOutput = args[6]
    fTestOutput = args[7]
    featFlag = int(args[8])
    featDictInput = args[9]

    (trainLabel, trainText) = review_input(trainInput)
    (validLabel, validText) = review_input(validInput)
    (testLabel, testText) = review_input(testInput)

    if featFlag == 1:
        dictVocab = dict_input(dictInput)
        fmtTrain = model1(dictVocab, trainText, trainLabel)
        fmtValid = model1(dictVocab, validText, validLabel)
        fmtTest = model1(dictVocab, testText, testLabel)
        # feature vector for model 1 bag-of-words in stack of rows
        feat_output(fTrainOutput, fmtTrain, '%s')
        feat_output(fValidOutput, fmtValid, '%s')
        feat_output(fTestOutput, fmtTest, '%s')
    elif featFlag == 2:
        (featVocab, vocabWeight) = featdict_input(featDictInput)
        fmtTrain = model2(featVocab, vocabWeight, trainText, trainLabel)
        fmtValid = model2(featVocab, vocabWeight, validText, validLabel)
        fmtTest = model2(featVocab, vocabWeight, testText, testLabel)
        # feature vector for model 2 word2vec in stack of rows
        feat_output(fTrainOutput, fmtTrain, '%1.6f')
        feat_output(fValidOutput, fmtValid, '%1.6f')
        feat_output(fTestOutput, fmtTest, '%1.6f')

''' 
python feature.py largedata/train_data.tsv largedata/valid_data.tsv largedata/test_data.tsv dict.txt largeoutput/formatted_train.tsv largeoutput/formatted_valid.tsv largeoutput/formatted_test.tsv 2 word2vec.txt
python feature.py smalldata/train_data.tsv smalldata/valid_data.tsv smalldata/test_data.tsv dict.txt smalloutput/formatted_train.tsv smalloutput/formatted_valid.tsv smalloutput/formatted_test.tsv 1 word2vec.txt
'''
