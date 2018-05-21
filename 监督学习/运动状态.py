# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))
    for file in feature_paths:
        # 逗号分隔读取特征数据， 问号替换标记为缺失值， 不含表头
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        # 使用平均值补全缺失值， 然后将数据补全
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))

    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label


if __name__ == '__main__':
    ''' 数据路径 '''
    featurePaths = ['./data/dataset/A/A.feature', './data/dataset/B/B.feature', './data/dataset/C/C.feature', './data/dataset/D/D.feature', './data/dataset/E/E.feature']
    labelPaths = ['./data/dataset/A/A.label', './data/dataset/B/B.label', './data/dataset/C/C.label', './data/dataset/D/D.label', './data/dataset/E/E.label']
    ''' 读入数据  '''
    x_train, y_train = load_datasets(featurePaths[:4], labelPaths[:4])
    x_test, y_test = load_datasets(featurePaths[4:], labelPaths[4:])
    # 使用全量数据作为训练集， train_test_split函数打乱训练数据
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)

    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')

    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')

    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')

    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))

"""
 从准确度的角度衡量，贝叶斯分类器的效果最好
 从召回率和F1值的角度衡量，k近邻效果最好
 贝叶斯分类器和k近邻的效果好于决策树
"""
"""
The classification report for knn:
             precision    recall  f1-score   support

        0.0       0.56      0.60      0.58    102341
        1.0       0.92      0.93      0.93     23699
        2.0       0.94      0.78      0.85     26864
        3.0       0.83      0.82      0.82     22132
        4.0       0.85      0.88      0.87     32033
        5.0       0.39      0.21      0.27     24646
        6.0       0.77      0.89      0.82     24577
        7.0       0.80      0.95      0.87     26271
       12.0       0.32      0.33      0.33     14281
       13.0       0.16      0.22      0.19     12727
       16.0       0.90      0.67      0.77     24445
       17.0       0.89      0.96      0.92     33034
       24.0       0.00      0.00      0.00      7733

avg / total       0.69      0.69      0.68    374783



The classification report for DT:
             precision    recall  f1-score   support

        0.0       0.49      0.73      0.58    102341
        1.0       0.66      0.96      0.78     23699
        2.0       0.83      0.86      0.84     26864
        3.0       0.93      0.70      0.80     22132
        4.0       0.23      0.16      0.19     32033
        5.0       0.75      0.85      0.80     24646
        6.0       0.88      0.64      0.74     24577
        7.0       0.61      0.15      0.24     26271
       12.0       0.60      0.64      0.62     14281
       13.0       0.66      0.50      0.57     12727
       16.0       0.56      0.07      0.13     24445
       17.0       0.84      0.85      0.85     33034
       24.0       0.42      0.32      0.37      7733

avg / total       0.62      0.61      0.58    374783



The classification report for Bayes:
             precision    recall  f1-score   support

        0.0       0.62      0.81      0.70    102341
        1.0       0.97      0.91      0.94     23699
        2.0       1.00      0.65      0.79     26864
        3.0       0.60      0.66      0.63     22132
        4.0       0.91      0.77      0.83     32033
        5.0       1.00      0.00      0.00     24646
        6.0       0.87      0.72      0.79     24577
        7.0       0.31      0.47      0.37     26271
       12.0       0.52      0.59      0.55     14281
       13.0       0.61      0.50      0.55     12727
       16.0       0.89      0.72      0.79     24445
       17.0       0.75      0.91      0.82     33034
       24.0       0.59      0.24      0.34      7733

avg / total       0.74      0.68      0.67    374783
"""