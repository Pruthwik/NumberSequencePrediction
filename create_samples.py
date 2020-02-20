def createSequenceOfNumbers(n):
    sequenceOfNumbers = [([i + 1, i + 2, i + 3, i + 4], [i + 2, i + 3, i + 4, i + 5]) for i in range(n)]
    return sequenceOfNumbers


def writeListsToFile(filePath, dataList):
    individualSamples = [' '.join(map(str, sample)) for sample in dataList]
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(individualSamples) + '\n')


def main():
    n = 1996
    sequenceOfNumbers = createSequenceOfNumbers(n)
    inputList, outputList = list(zip(* sequenceOfNumbers))
    # print(inputList)
    writeListsToFile('inputNumbers.txt', inputList)
    writeListsToFile('outputNumbers.txt', outputList)


if __name__ == '__main__':
    main()
