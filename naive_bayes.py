from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
#加载数据集
iris = datasets.load_iris()

X = iris.data
y = iris.target

print('Sample num:', len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=142)

#特征缩放 start
#https://blog.csdn.net/kyriehe/article/details/77507473
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) #估算每个特征的平均值和标准差
print(sc.mean_)  #特征的平均值
print(sc.scale_)  #特征的标准差


clf = GaussianNB()
#训练模型
clf.fit(X_train, y_train)
#预测结果
ans = clf.predict(X_test)
#计算准确率
cnt = 0
for i in range(len(y_test)):
    if ans[i] - y_test[i] < 1e-1:
        cnt += 1
print('accurancy:', (cnt * 100.0 / len(y_test)), '%')
