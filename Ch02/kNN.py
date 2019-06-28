"""
Created on Sep 16, 2010, Modified on Jun. 26, 2019
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
@modified by: mengqiu
"""
import numpy as np
import operator
import os


def classify0(inX, dataSet, labels, k):
    """
    KNN for classification.
    :param inX:         test data
    :param dataSet:     training data
    :param labels:      labels
    :param k:           number of neighbors
    :return:
        sorted classes
    """
    # 1. Calculate distances
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 2. Take k neighbors
    sortedDistIndicies = distances.argsort()     
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. Sort class
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    """
    Create a simple training dataset.
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    """
    Transform txt file to a numpy matrix.
    :param filename:
    :return:
    """
    fr = open(filename)
    # 1. get the number of lines in the file
    lines = fr.readlines()
    numberOfLines = len(lines)
    # 2. prepare matrix to return
    returnMat = np.zeros((numberOfLines, 3))
    # 3. prepare labels return
    classLabelVector = []
    # 4. Read txt and set to numpy matrix
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    Normalization function.
    :param dataSet:     input dataset
    :return:
        normDataSet     normalized dataset
        ranges          normalization range
        minVals         minimum values of each column
    """
    # minimum value of each column
    minVals = dataSet.min(0)
    # maximum value of each column
    maxVals = dataSet.max(0)
    # calculate value range
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # element wise divide
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


def datingClassTest():
    """
    Classification test using kNN.
    """
    # hold out 10%
    hoRatio = 0.10
    # load data setfrom file
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :],
            datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("the total error case is: %d" % errorCount)


def classifyPerson():
    """
    Classify a person in given conditions.
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percenTats = float(input(
        "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    # load data setfrom file
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = np.array([ffMiles, percenTats, iceCream])
    classifierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, 3)

    print("You will probably like this person: ",
          resultList[classifierResult - 1])


def img2vector(filename):
    """
    transform text images to vector
    :param filename:    name of text files.
    :return:
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """
    Hand write classification.
    """
    hwLabels = []
    # load the training set
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    # iterate through the test set
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
