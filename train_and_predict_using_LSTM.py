from sys import argv
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
from keras.layers import TimeDistributed
import numpy as np


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))


def convertStringsToNumbers(listOfStrings):
    return [list(map(int, x.split())) for x in listOfStrings]


def createNumberToIndexDictionary(n):
    return {number: index + 1 for index, number in enumerate(range(1, n + 1))}


def createReverseDict(dataDict):
    return {value: key for key, value in dataDict.items()}


def convertDataIntoIndices(dataList, dataToIndexDict):
    return [[dataToIndexDict[i] for i in data] for data in dataList]


def trainModel(inputData, outputData, vocabSize):
    inputLayer = Input(shape=(4,))
    embeddingLayer = Embedding(vocabSize + 1, 100)(inputLayer)
    lstmLayer = LSTM(100, return_sequences=True)(embeddingLayer)
    outputLayer = TimeDistributed(Dense(vocabSize + 1, activation='softmax'))(lstmLayer)
    model = Model(inputs=inputLayer, outputs=outputLayer)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    model.fit(inputData, outputData, epochs=50)
    return model


def main():
    '''
    takes 4 command line arguments
    first 2 args are input number sequences and output sequences
    3rd argument is the file where the model will be saved
    4th argument is the file where the predictions will be saved
    '''
    np.random.seed(100)
    inputFile = argv[1]
    inputLines = readLinesFromFile(inputFile)
    listOfInputNumbers = convertStringsToNumbers(inputLines)
    outputFile = argv[2]
    modelFile = argv[3]
    predictedFile = argv[4]
    totalNumbers = 2000
    outputLines = readLinesFromFile(outputFile)
    listOfOutputNumbers = convertStringsToNumbers(outputLines)
    numberToIndexDict = createNumberToIndexDictionary(totalNumbers)
    indexToNumberDict = createReverseDict(numberToIndexDict)
    inputIndices = convertDataIntoIndices(listOfInputNumbers, numberToIndexDict)
    outputIndices = convertDataIntoIndices(listOfOutputNumbers, numberToIndexDict)
    inputIndices = np.array(inputIndices)
    outputIndices = np.array(outputIndices)
    outputIndicesOneHot = np.zeros((outputIndices.shape[0], outputIndices.shape[1], totalNumbers + 1))
    for i in range(outputIndicesOneHot.shape[0]):
        for j in range(outputIndicesOneHot.shape[1]):
            outputIndicesOneHot[i, j, outputIndices[i][j]] = 1
    model = trainModel(inputIndices, outputIndicesOneHot, totalNumbers)
    model.save(modelFile)
    testSamples = [[1, 2, 3, 8], [100, 202, 301, 12], [99, 111, 888, 928], [1501, 1555, 1787, 1989]]
    testIndices = convertDataIntoIndices(testSamples, numberToIndexDict)
    testIndices = np.array(testIndices)
    print(testIndices.shape)
    predictedTest = model.predict(testIndices)
    predictedIndices = list()
    for i in range(predictedTest.shape[0]):
        print('Sample', i + 1)
        tempList = list()
        for j in range(predictedTest.shape[1]):
            print('time step', j + 1, np.argmax(predictedTest[i][j]))
            tempList.append(str(indexToNumberDict[np.argmax(predictedTest[i][j])]))
        predictedIndices.append(' '.join(tempList))
    writeListToFile(predictedFile, predictedIndices)


if __name__ == '__main__':
    main()
