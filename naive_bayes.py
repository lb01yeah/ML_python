from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

print('Sample num:', len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = GaussianNB()

clf.fit(X_train, y_train)

ans = clf.predict(X_test)

cnt = 0
for i in range(len(y_test)):
    if ans[i] - y_test[i] < 1e-1:
        cnt +=1
print('accurancy:', (cnt * 100.0 / len(y_test)), '%')
