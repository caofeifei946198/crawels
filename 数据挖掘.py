# import requests
# url = "http://aima.cs.berkeley.edu/data/iris.csv"
# res = requests.get(url)
# localFile = open('iris.csv','w')
# localFile.write(res.text)
# localFile.close()

from numpy import genfromtxt,zeros
#reaa the first 4 colums
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
#print(data.shape)
#read the fifth columns
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)
#print(target.shape)
#establish a collection of unique elements
#print(set(target))
from pylab import plot,show
# plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
# plot(data[target=='versicolo',0],data[target=='versicolo',2],'ro')
# plot(data[target=='virginica',0],data[target=='virginica',2],'go')
# show()
from pylab import figure,subplot,hist,xlim,show
# xmin = min(data[:,0])
# xmax = max(data[:,0])
# figure()
# subplot(411)#distribution of the setosa class
# hist(data[target=='setosa',0],color='b',alpha=.7)
# xlim(xmin,xmax)
# subplot(412)
# hist(data[target=='versicolo',0],color='r',alpha=.7)
# xlim(xmin,xmax)
# subplot(413)
# hist(data[target=='virginica',0],color='g',alpha=.7)
# xlim(xmin,xmax)
# subplot(414)
# hist(data[:,0],color='y',alpha=.7)
# xlim(xmin,xmax)
# show()

#class 把字符串转换成整型数据
# t = zeros(len(target))
#
# t[target == 'setosa'] = 1
# t[target == 'versicolo'] = 2
# t[target == 'virginica'] = 3
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(data,t)#training on the iris dataset
# print(classifier.predict(data[0]))
# print(t[0])
#使用交叉验证测试分类器
# from sklearn import cross_validation
# train, test, t_train, t_test = cross_validation.train_test_split(
#     data, t, test_size=0.4, random_state=0
# )
# classifier.fit(train,t_train)#train
# print(classifier.score(test,t_test))#test
# #使用混淆矩阵测试分类器
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(classifier.predict(test),t_test))
# from sklearn.metrics import classification_report
# print(classification_report(classifier.predict(test),t_test,target_names=
# ['setosa','versicolo','virginica']))
#s使用更加精确的模型cross validation,即多次将数据分为不同的训练集和测试集最终分类器评估选取多次
#预测的平均值
# from sklearn.cross_validation import cross_val_score
# #cross validation with 6 iterations
# scores = cross_val_score(classifier,data,t,cv=6)
# print(scores)
# from numpy import mean
# print(mean(scores))
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, init='random')#initialization
# kmeans.fit(data)#actual execution
# c = kmeans.predict(data)
# from sklearn.metrics import completeness_score,homogeneity_score
# print(completeness_score(t,c))
# print(homogeneity_score(t,c))
# from pylab import plot,show
# figure()
# subplot(211)
# plot(data[t==1,0],data[t==1,2],'bo')
# plot(data[t==2,0],data[t==2,2],'ro')
# plot(data[t==3,0],data[t==3,2],'go')
# subplot(212)
# plot(data[c==1,0],data[c==1,2],'bo',alpha=.7)
# plot(data[c==2,0],data[c==2,2],'go',alpha=.7)
# plot(data[c==0,0],data[c==0,2],'mo',alpha=.7)
# show()
# from numpy.random import rand
# x = rand(40,1)#explanatory variable
# y = x*x*x+rand(40,1)/5#dependent variable
# from sklearn.linear_model import LinearRegression
# linreg = LinearRegression()
# linreg.fit(x,y)
# from numpy import linspace,matrix
# xx = linspace(0,1,40)
# plot(x,y,'o',xx,linreg.predict(matrix(xx).T),'--r')
# show()
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(linreg.predict(x),y))#当拟合线很完美时值为临
# from numpy import corrcoef
# corr = corrcoef(data.T)#Ｔgives the transpose
# print(corr)
# from pylab import pcolor,colorbar,xticks,yticks
# from numpy import arange
# pcolor(corr)
# colorbar()#add
# #arange the names of the variable on the axis
# xticks(arange(0.5,4.5),['sepal length','sepal width','petal length','petal width'])#,notation=-20)
# yticks(arange(0.5,4.5),['sepal length','sepal width','petal length','petal width'])#,notation=-20)
# show()
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pcad = pca.fit_transform(data)
# plot(pcad[target=='setosa',0],pcad[target=='setosa',1],'bo')
# plot(pcad[target=='versicolo',0],pcad[target=='versicolo',1],'ro')
# plot(pcad[target=='virginica',0],pcad[target=='virginica',1],'go')
# print(pca.explained_variance_ratio_)#判断主成分包含的信息量
# print(1-sum(pca.explained_variance_ratio_))#输出在转化和＝过程中损失的信息量
# data_inv = pca.inverse_transform(pcad)
# print(abs(sum(sum(data-data_inv))))#估算逆变换的结果和原始数据的相似度
# for i in range(1,5):
#     pca = PCA(n_components=i)
#     pca.fit(data)
#     print(sum(pca.explained_variance_ratio_) * 100,'%')
#     #可以看出只要使用三个ｐｃａ就可以把汗原味化几乎１００的信息
import requests
url = "https://gephi.org/datasets/lesmiserables.gml.zip"
res = requests.get(url)
localFile = open('lesmiserables.gml','w')
localFile.write(res.text)
localFile.close()
 