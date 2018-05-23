# -*-coding:utf-8 -*-
import numpy as np
from os import listdir
from sklearn import neighbors


# 加载训练数据
def img2vector(fileName):
    retMat = np.zeros([1024], int)  # 定义返回的矩阵，大小为1*1024
    fr = open(fileName)  # 打开包含32*32大小的数字文件
    lines = fr.readlines()  # 读取文件的所有行
    for i in range(32):  # 遍历文件所有行
        for j in range(32):  # 并将01数字存放在retMat中
            retMat[i * 32 + j] = lines[i][j]
    return retMat

def readDataSet(path):
    fileList = listdir(path)  # 获取文件夹下的所有文件
    numFiles = len(fileList)  # 统计需要读取的文件的数目
    dataSet = np.zeros([numFiles, 1024], int)  # 用于存放所有的数字文件
    hwLabels = np.zeros([numFiles])  # 用于存放对应的标签(与神经网络的不同)
    for i in range(numFiles):  # 遍历所有的文件
        filePath = fileList[i]  # 获取文件名称/路径
        digit = int(filePath.split('_')[0])  # 通过文件名获取标签
        hwLabels[i] = digit  # 直接存放数字，并非one-hot向量
        dataSet[i] = img2vector(path + '/' + filePath)  # 读取文件内容
    return dataSet, hwLabels


train_dataSet, train_hwLabels = readDataSet('./data/handwrite/digits/trainingDigits')

# 构建KNN分类器
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)