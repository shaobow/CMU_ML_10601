import matplotlib.pyplot as plt

from learnhmm import word2index
from learnhmm import tag2index
from learnhmm import count_init
from learnhmm import count_emit
from learnhmm import count_trans
from forwardbackward import predict


def input_labeled_text_number(textdir, num):
    # define an empty list
    observes = []
    states = []
    currobs = []
    currsts = []
    # open file and read the content in a list
    with open(textdir, 'r') as infile:
        count = 0
        for line in infile:
            if line[0] != '\n':
                if line[-1] != '\n':  # the last line
                    [ob, st] = line.split('\t')
                    currobs.append(ob)
                    currsts.append(st)
                    observes.append(currobs)
                    states.append(currsts)
                    break
                else:
                    [ob, st] = line[:-1].split('\t')
                    currobs.append(ob)
                    currsts.append(st)

            elif line[0] == '\n':  # new instance
                observes.append(currobs)
                states.append(currsts)
                currobs = []
                currsts = []
                count += 1
                if count > num:
                    break
    return observes, states


trainInputDir = "en_data/train.txt"
index2wordDir = "en_data/index_to_word.txt"
index2tagDir = "en_data/index_to_tag.txt"
validInputDir = "en_data/validation.txt"

numSeq = [10, 100, 1000, 10000]
trainLL = []
validLL = []
for ns in numSeq:
    trainObserves, trainStates = input_labeled_text_number(trainInputDir, ns)
    trainObservesIndex, numWord = word2index(trainObserves, index2wordDir)
    trainStatesIndex, numTag = tag2index(trainStates, index2tagDir)
    validObserves, validStates = input_labeled_text_number(validInputDir, ns)
    validObservesIndex, numWord = word2index(validObserves, index2wordDir)
    validStatesIndex, numTag = tag2index(validStates, index2tagDir)

    trainInit = count_init(trainStatesIndex, numTag)
    trainEmit = count_emit(trainStatesIndex, trainObservesIndex, numTag, numWord)
    trainTrans = count_trans(trainStatesIndex, numTag)

    _, validLogLikelihood = predict(validObservesIndex, trainInit, trainEmit, trainTrans, index2tagDir)
    _, trainLogLikelihood = predict(trainObservesIndex, trainInit, trainEmit, trainTrans, index2tagDir)

    trainLL.append(trainLogLikelihood)
    validLL.append(validLogLikelihood)

print(trainLL)
print(validLL)

plt.figure(1, dpi=200)
plt.plot(numSeq, trainLL, label='train')
plt.plot(numSeq, validLL, label='valid')
plt.legend()
plt.xlabel('number of sequences')
plt.ylabel('average log likelihood')
plt.show()


