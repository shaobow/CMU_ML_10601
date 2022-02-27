import sys
import csv 
import numpy as np

if __name__=='__main__':
    args = sys.argv
    assert(len(args) == 7)
    trainInput=args[1]
    testInput=args[2]
    maxDepth=int(args[3])
    trainOut=args[4]
    testOut=args[5]
    metricsOut=args[6]

    def tsv2ArrayWithoutHeader(fileName):
        fileDataArray=np.genfromtxt(fileName,dtype=str,delimiter='\t',skip_header=1)
        return fileDataArray
    def readheader(fileName):
        fileHeaderArray=np.loadtxt(fileName,dtype=str,delimiter='\t')
        fileHeaderArray=fileHeaderArray[[0],...]
        return fileHeaderArray

    def outputFile(fileName,outputContent):
        with open(fileName, 'w') as fileOut:
            np.savetxt(fileName,outputContent,fmt='%s')

    def outputTxt(fileName,outputContent):
        with open(fileName, 'w') as fileOut:
            fileOut.write(outputContent)
        
    def getEntropy(dataArrayY):
        dataSize=np.size(dataArrayY)
        dataVal=np.unique(dataArrayY)
        
        if len(dataVal)==1:
            dataEntropy=0
            return dataEntropy
        else:
            dataEntropy=0            
            for i in range(len(dataVal)):
                dataValNum_i=np.count_nonzero(dataArrayY==dataVal[i])
                dataValProb_i=dataValNum_i/dataSize
                dataEntropy-=dataValProb_i*np.log2(dataValProb_i)                
            return dataEntropy
                
    def getCondEntropy(dataArrayY,dataArrayX):
        dataSizeY=np.size(dataArrayY)
        dataSizeX=np.size(dataArrayX)        
        dataValY=np.unique(dataArrayY)
        dataValX=np.unique(dataArrayX)
        
        if len(dataValY)==1:
            dataEntropyY_X=0
            return dataEntropyY_X
        elif  len(dataValX)==1:
            dataEntropyY_X=getEntropy(dataArrayY)
            return dataEntropyY_X
        else:
            dataEntropyY_X=0            
            for i in range(len(dataValX)):
                dataValNumXi=np.count_nonzero(dataArrayX==dataValX[i])
                dataValProbXi=dataValNumXi/dataSizeX                                
                dataArrayY_Xi=dataArrayY[np.argwhere(dataArrayX==dataValX[i])]
                dataEntropyY_Xi=getEntropy(dataArrayY_Xi)                    
                dataEntropyY_X+=dataValProbXi*dataEntropyY_Xi            
            return dataEntropyY_X
                
    def getMutualInfo(dataArrayY,dataArrayX):        
        dataEntropyY=getEntropy(dataArrayY)
        dataEntropyY_X=getCondEntropy(dataArrayY,dataArrayX)
        dataMutInfoYX=dataEntropyY-dataEntropyY_X        
        return dataMutInfoYX
    
    
    def majorityVote(inputDataArray):
        labelIdx=np.size(inputDataArray,axis=1)-1
        labelColumn=inputDataArray[:,labelIdx]
        labelValue=np.unique(inputDataArray[:,labelIdx])
        if len(labelValue)==2:
            firstLabelNum=np.count_nonzero(labelColumn==labelValue[0])
            secondLabelNum=np.count_nonzero(labelColumn==labelValue[1])
            if firstLabelNum>secondLabelNum:
                return labelValue[0]
            elif firstLabelNum==secondLabelNum:
                lexiList=sorted([labelValue[0],labelValue[1]])
                return lexiList[1]
            else:
                return labelValue[1]
        elif len(labelValue)==1:
            return labelValue[0]

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

    def getSplitAttribute(inputDataArray):
        attributeNum=np.size(inputDataArray,axis=1)-1
        mutInfoArray=np.zeros((attributeNum,1))
        labelDataArray=inputDataArray[:,attributeNum]
        for i in range(attributeNum):
            attributeDataArray=inputDataArray[:,i]
            mutInfoArray[i]=getMutualInfo(labelDataArray,attributeDataArray)
        if np.amax(mutInfoArray)==0:
            return None
        else:
            maxMIAttributeIdx=np.argmax(mutInfoArray) # split on first attribute to break ties
            return maxMIAttributeIdx 
                                    
    def trainDecisionStump(inputNode,maxDepth):
        if inputNode.depth>maxDepth:
            # inputNode.data.shape (57,8)
            label=majorityVote(inputNode.data)
            inputNode.predict=label
            #print("maxDepth")
            return 
        elif inputNode.attribute==None: # possible for subset to be attribute-pure before use up all 
            label=majorityVote(inputNode.data)
            inputNode.predict=label
            #print("zero mutualinfo")
            return
        elif len(inputNode.usedAttribute)==len(inputNode.data[0,:]):
            label=majorityVote(inputNode.data)
            inputNode.predict=label            
            #print("use up attribute")
            return
        else:
            attributeArray=inputNode.data[...,[inputNode.attribute]] # when use : get 1-D data
            attributeVal=np.unique(attributeArray)
            inputNode.attributeVal=attributeVal
            
            leftNodeIdx=np.argwhere(attributeArray==attributeVal[0])
            rightNodeIdx=np.argwhere(attributeArray==attributeVal[1])
            
            leftDataArray=inputNode.data[leftNodeIdx[...,0]]
            leftLabelVal=np.unique(leftDataArray[...,-1])
            
            if len(leftLabelVal)==2:
                leftLabelNum_1=np.count_nonzero(leftDataArray[...,[-1]]==leftLabelVal[0])
                leftLabelNum_2=np.count_nonzero(leftDataArray[...,[-1]]==leftLabelVal[1])
            else:
                leftLabelNum_1=np.count_nonzero(leftDataArray[...,[-1]]==leftLabelVal[0])
                leftLabelNum_2=0
            inputNode.prtLNum1=leftLabelNum_1
            inputNode.prtLNum2=leftLabelNum_2
            
            rightDataArray=inputNode.data[rightNodeIdx[...,0]]
            rightLabelVal=np.unique(rightDataArray[...,-1])

            if len(rightLabelVal)==2:
                rightLabelNum_1=np.count_nonzero(rightDataArray[...,[-1]]==rightLabelVal[0])
                rightLabelNum_2=np.count_nonzero(rightDataArray[...,[-1]]==rightLabelVal[1])
            else:
                rightLabelNum_2=np.count_nonzero(rightDataArray[...,[-1]]==rightLabelVal[0])
                rightLabelNum_1=0 
            inputNode.prtRNum1=rightLabelNum_1
            inputNode.prtRNum2=rightLabelNum_2
            
            inputNode.addLeft(leftDataArray,inputNode.depth+1,inputNode.usedAttribute)
            inputNode.addRight(rightDataArray,inputNode.depth+1,inputNode.usedAttribute)
            
            trainDecisionStump(inputNode.left,maxDepth)
            trainDecisionStump(inputNode.right,maxDepth)
                
    def trainDecisionTree(inputDataArray,maxDepth):        
        rootNode=dtNode(inputDataArray,1,np.zeros((0,1),dtype=str))
        trainDecisionStump(rootNode,maxDepth)
        return rootNode
    

    def decisionStump(inputNode,inputDataRow):           
        if inputNode.predict!=None:
            label=inputNode.predict
            return label
        else:
            attributeIdx=inputNode.attribute
            if inputDataRow[[0],[attributeIdx]]==inputNode.attributeVal[0]:
                label=decisionStump(inputNode.left,inputDataRow)
            else:
                label=decisionStump(inputNode.right,inputDataRow)
            return label
    
    def predictLabel(inputNode,inputDataArray):
        predictNum=np.size(inputDataArray,axis=0)
        columnNum=np.size(inputDataArray,axis=1)
        predictArray=np.zeros((predictNum,1),dtype=object)
        for i in range(predictNum):
            inputDataRow=inputDataArray[i:i+1,0:columnNum]
            predictArray[i]=decisionStump(inputNode,inputDataRow)
        return predictArray
    
    def drawRecur(inputNode,fileHeaderArray,maxDepth):
        if inputNode.depth>maxDepth:
            return
        elif inputNode.attribute!=None:
            labelArray=inputNode.data[...,-1]
            attributeArray=inputNode.data[...,inputNode.attribute]
            labelVal=np.unique(labelArray)
            
            print("|"*inputNode.depth+" "+fileHeaderArray[0,inputNode.attribute]+" = "+inputNode.attributeVal[0]+f": [{inputNode.prtLNum1} "+labelVal[0]+f"/{inputNode.prtLNum2} "+labelVal[1]+"]")
            drawRecur(inputNode.left,fileHeaderArray,maxDepth)
            print("|"*inputNode.depth+" "+fileHeaderArray[0,inputNode.attribute]+" = "+inputNode.attributeVal[1]+f": [{inputNode.prtRNum1} "+labelVal[0]+f"/{inputNode.prtRNum2} "+labelVal[1]+"]")
            drawRecur(inputNode.right,fileHeaderArray,maxDepth)

                    
    def prettyPrint(inputNode,fileHeaderArray,maxDepth):
        labelArray=inputNode.data[...,-1]
        labelVal=np.unique(labelArray)
        firstLabelNum=np.count_nonzero(labelArray==labelVal[0])
        secondLabelNum=np.count_nonzero(labelArray==labelVal[1])
        print(f"\n[{firstLabelNum} "+labelVal[0]+"/"+f"{secondLabelNum} "+labelVal[1]+"]")
        drawRecur(inputNode,fileHeaderArray,maxDepth)
        
    class dtNode(object):
        def __init__(self,nodeDataArray,nodeDepth,previousAttribute):
            self.data=nodeDataArray
            self.depth=nodeDepth
            self.dataNum=np.size(nodeDataArray,axis=0)
            self.attribute=getSplitAttribute(self.data) # index of split attribute
            self.usedAttribute=np.append(previousAttribute,self.data[0,self.attribute])
            self.attributeVal=None
            self.predict=None
            self.left=None
            self.right=None
            self.prtLNum1=0
            self.prtLNum2=0
            self.prtRNum1=0
            self.prtRNum2=0
        def addLeft(self,leftDataArray,leftDepth,previousAttribute):
            leftNode=dtNode(leftDataArray,leftDepth,previousAttribute)
            self.left=leftNode
        def addRight(self,rightDataArray,rightDepth,previousAttribute):
            rightNode=dtNode(rightDataArray,rightDepth,previousAttribute)
            self.right=rightNode        

    trainDataArray=tsv2ArrayWithoutHeader(trainInput) # <U10 <class 'numpy.ndarray'>
    testDataArray=tsv2ArrayWithoutHeader(testInput)
    fileHeaderArray=readheader(trainInput)
    
    trainedDT=trainDecisionTree(trainDataArray,maxDepth)
    
    trainDataPrediction=predictLabel(trainedDT,trainDataArray)
    testDataPrediction=predictLabel(trainedDT,testDataArray)
   
    generalMetrics=(f"error(train): {errorRate(trainDataArray,trainDataPrediction)}"+"\n"+f"error(test): {errorRate(testDataArray,testDataPrediction)}")

    
    outputFile(trainOut,trainDataPrediction)
    outputFile(testOut,testDataPrediction)
    outputTxt(metricsOut,generalMetrics)
    
    prettyPrint(trainedDT,fileHeaderArray,maxDepth)

    attributeNum=np.size(trainDataArray,axis=1)
    print(attributeNum)




# python decisionTree.py education_train.tsv education_test.tsv 3 edu_3_train.labels edu_3_test.labels edu_3_metrics.txt
# python decisionTree.py small_train.tsv small_test.tsv 2 small_2_train.labels small_2_test.labels small_2_metrics.txt
# python decisionTree.py politicians_train.tsv politicians_test.tsv 3 pol_3_train.labels pol_3_test.labels pol_3_metrics.txt
# decisionTree.py politicians_train.tsv politicians_test.tsv 6 train.labels test.labels metrics.txt
# python decisionTree.py education_train.tsv education_test.tsv 4 edu_4_train.labels edu_4_test.labels edu_4_metrics.txt
# python decisionTree.py mushroom_train.tsv mushroom_test.tsv 4 mushroom_4_train.labels mushroom_4_test.labels mushroom_4_metrics.txt

# python decisionTree.py mushroom_train.tsv mushroom_test.tsv 20 mushroom_20_train.labels mushroom_20_test.labels mushroom_20_metrics.txt