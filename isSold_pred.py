import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
#prepare data
test_set = pd.read_csv('TestSet.csv')
train_set = pd.read_csv('TrainingSet.csv')
train = train_set.drop(["EbayID", 'QuantitySold', "SellerName"],axis=1)
train_target = train_set['QuantitySold']
n_trainSamples,n_features = train.shape

#训练过程中SGDclassifier利用不同的mini_batch学习的效果
def plot_learning(clf,title):
    plt.figure()
    #记录上一次预测结果在本次batch中的预测结果
    validationScore = []
    #记录机上本次batch训练结果后的预测情况
    trainScore = []
    #最小训练批数
    mini_batch = 1000
    for idx in range(int(np.ceil(n_trainSamples / mini_batch))):
        x_batch = train[idx * mini_batch: min((idx + 1)
            * mini_batch, n_trainSamples)]
        y_batch = train_target[idx * mini_batch: min((idx + 1) * mini_batch,
             n_trainSamples)]
        if idx > 0:
            validationScore.append(clf.score(x_batch, y_batch))
        clf.partial_fit(x_batch, y_batch, classes = range(5))
        if idx < 0:
            trainScore.append(clf.score(x_batch, y_batch))
    plt.plot(trainScore, label="train score")
    plt.plot(validationScore, label="validation score")
    plt.xlabel("Mini batch")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.title(title)

#对数据进行归一化
scaler = StandardScaler()
train = scaler.fit_transform(train)
#创建SGDclassifier
clf = SGDClassifier(penalty='l2', alpha=0.001)
plot_learning(clf,'SGDClassifier')
plt.show()


