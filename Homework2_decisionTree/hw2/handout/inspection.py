import sys
import csv 
import numpy as np

if __name__=='__main__':
    args = sys.argv
    assert(len(args) == 3)
    dataInput=args[1]
    resultOutput=args[2]

    def tsv2ArrayWithoutHeader(fileName):
        fileDataArray=np.genfromtxt(fileName,dtype=str,delimiter='\t',skip_header=1)
        return fileDataArray

    def outputTxt(fileName,outputContent):
        with open(fileName, 'w') as fileOut:
                fileOut.write(outputContent)

    def majorityVote(inputSet):
        labelIdx=np.size(inputSet,axis=1)-1
        labelValue=np.unique(inputSet[:,labelIdx])
        #print(labelValue,labelValue[0],type(labelValue),type(labelValue[0]))
        if len(labelValue)==2:
            firstLabelNum=np.count_nonzero(inputSet[:,labelIdx]==labelValue[0])
            secondLabelNum=np.count_nonzero(inputSet[:,labelIdx]==labelValue[1])
            if firstLabelNum>secondLabelNum:
                return labelValue[0]
            elif firstLabelNum==secondLabelNum:
                lexiList=sorted([labelValue[0],labelValue[1]])
                return lexiList[1]                
            else:
                return labelValue[1]
        elif len(labelValue)==1:
            return labelValue[0]
        
    def getEntropy(inputSet):
        labelIdx=np.size(inputSet,axis=1)-1
        labelValue=np.unique(inputSet[:,labelIdx])
        if len(labelValue)==2:
            firstLabelNum=np.count_nonzero(inputSet[:,labelIdx]==labelValue[0])
            secondLabelNum=np.count_nonzero(inputSet[:,labelIdx]==labelValue[1])
            firstLabelProb=firstLabelNum/np.size(inputSet,axis=0)
            secondLabelProb=secondLabelNum/np.size(inputSet,axis=0)
            dataEntropy=-(firstLabelProb*np.log2(firstLabelProb)+secondLabelProb*np.log2(secondLabelProb))
            return dataEntropy
        elif len(labelValue)==1:
            dataEntropy=0
            return dataEntropy       
        
    def errorRate(inputSet,predictLabel):
        labelIdx=np.size(inputSet,axis=1)-1
        totalNum=np.size(inputSet,axis=0)
        realLabel=inputSet[:,labelIdx]
        erNum=0
        for i in range(totalNum):
            if realLabel[i]!=predictLabel:
                erNum+=1
        erRate=erNum/totalNum
        return erRate

    inputDataArray=tsv2ArrayWithoutHeader(dataInput)

    dataInspection=(f"entropy: {getEntropy(inputDataArray)}"+"\n"+f"error: {errorRate(inputDataArray,majorityVote(inputDataArray))}")

    outputTxt(resultOutput,dataInspection)


    # python inspection.py small_train.tsv small_inspect.txt
    # python inspection.py politicians_test.tsv politicians_test_inspect.txt