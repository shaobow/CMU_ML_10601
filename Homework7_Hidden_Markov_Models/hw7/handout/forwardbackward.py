import sys
import numpy as np
from learnhmm import input_labeled_text
from learnhmm import word2index
from learnhmm import tag2index


# write prediction into file
def output_prediction(predictdir, observes, predictions):
    first_inst = True
    first_line = True
    with open(predictdir, 'w') as outfile:
        for obs, prds in zip(observes, predictions):
            if first_inst:
                for ob, prd in zip(obs, prds):
                    if first_line:
                        outfile.write(f"{ob}\t{prd}")
                        first_line = False
                    else:
                        outfile.write(f"\n{ob}\t{prd}")
                first_inst = False
            else:
                outfile.write("\n")
                for ob, prd in zip(obs, prds):
                    outfile.write(f"\n{ob}\t{prd}")


def log_sum_exp(vec):
    m = np.max(vec, axis=0)
    return m+np.log(np.sum(np.exp(vec-m), axis=0))


def predict(observes, init, emit, trans, idxdir):
    dic = list(np.loadtxt(idxdir, dtype='str'))
    num_tag = len(init)
    predict_label = []
    loglikelihood = 0
    count = 0
    for obs in observes:
        count += 1
        num_obs = len(obs)
        alpha_log = np.zeros([num_tag, num_obs])
        beta_log = np.zeros([num_tag, num_obs])
        for t in np.arange(num_obs):
            if t == 0:
                for st in np.arange(num_tag):
                    alpha_log[[st], [0]] = np.log(emit[[st], [obs[0]]]) + np.log(init[[st], [0]])
                    beta_log[[st], [num_obs-1]] = 0
            else:
                for st in np.arange(num_tag):
                    v_alpha = alpha_log[:, [t - 1]] + np.log(trans[:, [st]])
                    alpha_log[[st], [t]] = np.log(emit[[st], [obs[t]]]) + log_sum_exp(v_alpha)
                    v_beta = beta_log[:, [num_obs-t]] + np.log(emit[:, [obs[num_obs-t]]]) + np.log(trans[[st], :].T)
                    beta_log[[st], [num_obs-1-t]] = log_sum_exp(v_beta)
        predict_idx = list(np.argmax(alpha_log+beta_log, axis=0))
        predict_label.append([(dic[idx]) for idx in predict_idx])
        loglikelihood += log_sum_exp(alpha_log[:, [-1]])[0]
    loglikelihood /= count
    return predict_label, loglikelihood


if __name__ == '__main__':
    args = sys.argv
    assert (len(args) == 9)
    validInputDir = args[1]
    index2wordDir = args[2]
    index2tagDir = args[3]
    hmmInitDir = args[4]
    hmmEmitDir = args[5]
    hmmTransDir = args[6]
    predOutputDir = args[7]
    metricOutputDir = args[8]

    '''Debug use dir'''
    # validInputDir = "toy_data/validation.txt"
    # index2wordDir = "toy_data/index_to_word.txt"
    # index2tagDir = "toy_data/index_to_tag.txt"
    # hmmInitDir = "toy_output/hmminit_mine.txt"
    # hmmEmitDir = "toy_output/hmmemit_mine.txt"
    # hmmTransDir = "toy_output/hmmtrans_mine.txt"
    # predOutputDir = "toy_output/predicted_mine.txt"
    # metricOutputDir = "toy_output/metrics_mine.txt"
    '''-------------'''
    # validInputDir = "en_data/validation.txt"
    # index2wordDir = "en_data/index_to_word.txt"
    # index2tagDir = "en_data/index_to_tag.txt"
    # hmmInitDir = "en_output/hmminit_mine.txt"
    # hmmEmitDir = "en_output/hmmemit_mine.txt"
    # hmmTransDir = "en_output/hmmtrans_mine.txt"
    # predOutputDir = "en_output/predicted_mine.txt"
    # metricOutputDir = "en_output/metrics_mine.txt"
    '''-------------'''
    # validInputDir = "fr_data/validation.txt"
    # index2wordDir = "fr_data/index_to_word.txt"
    # index2tagDir = "fr_data/index_to_tag.txt"
    # hmmInitDir = "fr_output/hmminit_mine.txt"
    # hmmEmitDir = "fr_output/hmmemit_mine.txt"
    # hmmTransDir = "fr_output/hmmtrans_mine.txt"
    # predOutputDir = "fr_output/predicted_mine.txt"
    # metricOutputDir = "fr_output/metrics_mine.txt"
    '''-------------'''

    validObserves, validStates = input_labeled_text(validInputDir)
    validObservesIndex, numWord = word2index(validObserves, index2wordDir)
    validStatesIndex, numTag = tag2index(validStates, index2tagDir)

    hmmInit = np.loadtxt(hmmInitDir, ndmin=2)
    hmmEmit = np.loadtxt(hmmEmitDir, ndmin=2)
    hmmTrans = np.loadtxt(hmmTransDir, ndmin=2)

    validPredictions, validLogLikelihood = predict(validObservesIndex, hmmInit, hmmEmit, hmmTrans, index2tagDir)

    validAccuracy = 0
    count = 0
    for prds, sts in zip(validPredictions, validStates):
        for prd, st in zip(prds, sts):
            count += 1
            if prd == st:
                validAccuracy += 1
    validAccuracy /= count

    with open(metricOutputDir, 'w') as otf:
        otf.write(f"Average Log-Likelihood: {validLogLikelihood}\n")
        otf.write(f"Accuracy: {validAccuracy}\n")
    output_prediction(predOutputDir, validObserves, validPredictions)

# python forwardbackward.py toy_data/validation.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_output/hmminit_mine.txt toy_output/hmmemit_mine.txt toy_output/hmmtrans_mine.txt toy_output/predicted_mine.txt toy_output/metrics_mine.txt
