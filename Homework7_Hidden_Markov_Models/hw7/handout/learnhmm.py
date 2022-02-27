import sys
import numpy as np


#  read train/valid data and return observations and states in two lists
def input_labeled_text(textdir):
    # define an empty list
    observes = []
    states = []
    currobs = []
    currsts = []
    # open file and read the content in a list
    with open(textdir, 'r') as infile:
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
    return observes, states


def word2index(observes, idxdir):
    dic = list(np.loadtxt(idxdir, dtype='str'))
    observes_idx = [[(dic.index(ob)) for ob in obs] for obs in observes]
    return observes_idx, len(dic)


def tag2index(states, idxdir):
    dic = list(np.loadtxt(idxdir, dtype='str'))
    states_idx = [[(dic.index(st)) for st in sts] for sts in states]
    return states_idx, len(dic)


def count_init(states_idx, num_tag):
    init = np.zeros([num_tag, 1])
    for st in np.arange(num_tag):
        init[[st], [0]] = [sts[0] for sts in states_idx].count(st)
    init += 1
    init /= np.sum(init, axis=0)
    return init


def count_emit(states_idx, observes_idx, num_tag, num_word):
    emit = np.zeros([num_tag, num_word])
    for obs, sts in zip(observes_idx, states_idx):
        for ob, st in zip(obs, sts):
            emit[[st], [ob]] += 1
    emit += 1
    emit /= np.sum(emit, axis=1).reshape([num_tag, 1])
    return emit


def count_trans(states_idx, num_tag):
    trans = np.zeros([num_tag, num_tag])
    for sts in states_idx:
        for st, st_next in zip(sts, sts[1:]):
            trans[[st], [st_next]] += 1
    trans += 1
    trans /= np.sum(trans, axis=1).reshape([num_tag, 1])
    return trans


if __name__ == '__main__':
    args = sys.argv
    assert (len(args) == 7)
    trainInputDir = args[1]
    index2wordDir = args[2]
    index2tagDir = args[3]
    hmmInitDir = args[4]
    hmmEmitDir = args[5]
    hmmTransDir = args[6]

    '''Debug use dir'''
    # trainInputDir = "toy_data/train.txt"
    # index2wordDir = "toy_data/index_to_word.txt"
    # index2tagDir = "toy_data/index_to_tag.txt"
    # hmmInitDir = "toy_output/hmminit_mine.txt"
    # hmmEmitDir = "toy_output/hmmemit_mine.txt"
    # hmmTransDir = "toy_output/hmmtrans_mine.txt"
    '''-------------'''
    # trainInputDir = "en_data/train.txt"
    # index2wordDir = "en_data/index_to_word.txt"
    # index2tagDir = "en_data/index_to_tag.txt"
    # hmmInitDir = "en_output/hmminit_mine.txt"
    # hmmEmitDir = "en_output/hmmemit_mine.txt"
    # hmmTransDir = "en_output/hmmtrans_mine.txt"
    '''-------------'''
    # trainInputDir = "fr_data/train.txt"
    # index2wordDir = "fr_data/index_to_word.txt"
    # index2tagDir = "fr_data/index_to_tag.txt"
    # hmmInitDir = "fr_output/hmminit_mine.txt"
    # hmmEmitDir = "fr_output/hmmemit_mine.txt"
    # hmmTransDir = "fr_output/hmmtrans_mine.txt"
    '''-------------'''

    trainObserves, trainStates = input_labeled_text(trainInputDir)
    trainObservesIndex, numWord = word2index(trainObserves, index2wordDir)
    trainStatesIndex, numTag = tag2index(trainStates, index2tagDir)

    trainInit = count_init(trainStatesIndex, numTag)
    trainEmit = count_emit(trainStatesIndex, trainObservesIndex, numTag, numWord)
    trainTrans = count_trans(trainStatesIndex, numTag)

    np.savetxt(hmmInitDir, trainInit, newline='\n')
    np.savetxt(hmmEmitDir, trainEmit, newline='\n')
    np.savetxt(hmmTransDir, trainTrans, newline='\n')

# python learnhmm.py toy_data/train.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_output/hmminit_mine.txt toy_output/hmmemit_mine.txt toy_output/hmmtrans_mine.txt
