"""
Created on Jun 28, 2019
Examples in book.
@created by: tianmengqiu
"""

import kNN
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # # Task 2.1.2
    # group, labels = kNN.createDataSet()
    # print(group)
    # print(labels)
    #
    # note = kNN.classify0([0, 0.5], group, labels, 3)
    # print(note)

    # # Task 2.2.1
    # datingDataMat, datingLabels = kNN.file2matrix("datingTestSet2.txt")
    # # # Task 2.2.2
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # # Figure 2-3
    # # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # # plt.sho2w()
    # # # Figure 2-4
    # # ax.scatter(
    # #     datingDataMat[:, 1], datingDataMat[:, 2],
    # #     15.0 * np.asarray(datingLabels), 15.0 * np.asarray(datingLabels))
    # # plt.show()
    # # Task 2.2.3
    # normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    # print(normMat)
    # print(ranges)
    # print(minVals)

    # # Task 2.2.4
    # kNN.datingClassTest()

    # # Task 2.2.5
    # kNN.classifyPerson()

    # Task 2.3.2
    kNN.handwritingClassTest()
