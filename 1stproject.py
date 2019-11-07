# -*- coding: utf-8 -*-
from numpy import set_printoptions
from sklearn import preprocessing
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
path = r"C:\Users\ASUS\Desktop\dataset\diabetes.csv"
data = read_csv(path)
array = data.values
X = array[:,0:8]
Y = array[:,8]
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict))


