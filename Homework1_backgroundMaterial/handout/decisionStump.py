import sys
import csv 
import numpy as np

if __name__=='__main__':
    args = sys.argv
    assert(len(args) == 7)
    trainInput=args[1]
    testInput=args[2]
    splitIndex=int(args[3])
    trainOut=args[4]
    testOut=args[5]
    metricsOut=args[6]

    def tsv2ArrayWithoutHeader(fileName):
        fileDataArray=np.genfromtxt(fileName,dtype=str,delimiter='\t',skip_header=1)
        return fileDataArray

#     def outputFile(fileName,outputContent):
#         with open(fileName, 'w') as fileOut:
#             for line in outputContent:
#                 fileOut.write(line+"\n")

    def outputFile(fileName,outputContent):
        with open(fileName, 'w') as fileOut:
            np.savetxt(fileName,outputContent,fmt='%s',delimiter='\t')
        
    def outputTxt(fileName,outputContent):
        with open(fileName, 'w') as fileOut:
                fileOut.write(outputContent)

    def majorityVote(inputSet):
        labelIdx=np.size(inputSet,axis=1)-1
        labelValue=np.unique(inputSet[:,labelIdx])
        if len(labelValue)==2:
            firstLabelNum=np.count_nonzero(inputSet[:,labelIdx]==labelValue[0])
            secondLabelNum=np.count_nonzero(inputSet[:,labelIdx]==labelValue[1])
            if firstLabelNum>=secondLabelNum:
                return labelValue[0]
            else:
                return labelValue[1]
        elif len(labelValue)==1:
            return labelValue[0]

    trainDataArray=tsv2ArrayWithoutHeader(trainInput)
    print(type(trainDataArray))
    testDataArray=tsv2ArrayWithoutHeader(testInput)
    
    def decisionStump(inputSet,attributeIdx):
        attributeColumn=inputSet[:,attributeIdx]
        attributeValue=np.unique(attributeColumn)
        leftNodeIdx=np.nonzero(attributeColumn==attributeValue[0])
        rightNodeIdx=np.nonzero(attributeColumn==attributeValue[1])
        
        leftNodeSet=inputSet[leftNodeIdx]
        rightNodeSet=inputSet[rightNodeIdx]
        leftNodeLabel=majorityVote(leftNodeSet)
        rightNodeLabel=majorityVote(rightNodeSet)
        predictionLabel=np.zeros((len(attributeColumn),1),dtype=object)
        
        predictionLabel[leftNodeIdx]=leftNodeLabel
        predictionLabel[rightNodeIdx]=rightNodeLabel
        
        predictionLabel.astype(str)
        
        print(predictionLabel,predictionLabel.dtype)

#         for i in leftNodeIdx[0]:
#             predictionLabel[i]=leftNodeLabel
#         for i in rightNodeIdx[0]:
#             predictionLabel[i]=rightNodeLabel
  
        return predictionLabel

    trainDataPrediction=decisionStump(trainDataArray,splitIndex)
    testDataPrediction=decisionStump(testDataArray,splitIndex)

    def errorRate(inputSet,predictLabel):
        labelIdx=np.size(inputSet,axis=1)-1
        totalNum=np.size(inputSet,axis=0)
        realLabel=inputSet[:,labelIdx]
        erNum=0
        for i in range(totalNum):
            if realLabel[i]!=predictLabel[i]:
                erNum+=1
        erRate=erNum/totalNum
        return erRate

    generalMetrics=(f"error(train): {errorRate(trainDataArray,trainDataPrediction):.6f}"+"\n"+f"error(test): {errorRate(testDataArray,testDataPrediction):.6f}")

    outputFile(trainOut,trainDataPrediction)
    outputFile(testOut,testDataPrediction)
    outputTxt(metricsOut,generalMetrics)


# python decisionStump.py politicians_train.tsv politicians_test.tsv 0 pol_0_train.labels pol_0_test.labels pol_0_metrics.txt