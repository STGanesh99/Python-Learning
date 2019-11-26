# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:49:59 2019

@author: ASUS
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
path = r'C:\Users\ASUS\Desktop\pims\diabetes.csv'
data = pd.read_csv(path)
data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['BMI'].replace(0,data['BMI'].mean())
data.drop(['Insulin'], axis = 1) 
x = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']]
y = data[['Outcome']]
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, shuffle=True)
classifiers = []
model1 = LogisticRegression()
classifiers.append(model1)
model2 = svm.SVC()
classifiers.append(model2)
model3 = tree.DecisionTreeClassifier()
classifiers.append(model3)
model4 = RandomForestClassifier()
classifiers.append(model4)
for clf in classifiers:
                          clf.fit(X_train, y_train)
                          y_pred= clf.predict(X_test)
                          acc = accuracy_score(y_test, y_pred)
                          print("Accuracy of %s is %s"%(clf, acc))
                          cm = confusion_matrix(y_test, y_pred)
                          print("Confusion Matrix of %s is %s"%(clf, cm))





