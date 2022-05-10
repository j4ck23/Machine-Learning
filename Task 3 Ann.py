import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import neural_network
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble

data = pd.read_csv("Task3 - dataset - HIV RVG.csv")#reads the data

#split the data 
x = data.iloc[:, 0:8]
y = data.select_dtypes(include=[object])

le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.10) #split the data to test train split woth a test size of 10%

#ANN function
tree = neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'sgd',
                                  max_iter = 10000, hidden_layer_sizes = (500, 500),
                                  activation = 'logistic',#Logic also applys sigmoid
                                  verbose = False, tol = 1e-10,
                                  random_state = 1)

tree.fit(X_train,y_train)#fits the daat to the ANN model

predict_tree = tree.predict(X_test)
print(classification_report(y_test,predict_tree))#Predicts using test daat on the trained ANN model and preduces a report to show accuracy

#Random forrest clasfiers with 5 and 10 leaf samples
forrest5 = ensemble.RandomForestClassifier(n_estimators=1000, min_samples_leaf=5)
forrest10 = ensemble.RandomForestClassifier(n_estimators=1000, min_samples_leaf=10)

forrest5.fit(X_train, y_train)
forrest10.fit(X_train,y_train)#fits the data

#predict the reults and preduce a report for accuracy.
predict_forrest5 = forrest5.predict(X_test)
predict_forrest10 = forrest10.predict(X_test)

print(classification_report(y_test,predict_forrest5))
print(classification_report(y_test,predict_forrest10))