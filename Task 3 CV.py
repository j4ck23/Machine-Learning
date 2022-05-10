import pandas as pd
import numpy as np
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import ensemble


data = pd.read_csv("Task3 - dataset - HIV RVG.csv")#reads the data

x = data.iloc[:, 0:8]
y = data.select_dtypes(include=[object])

le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)#sorts the data to required inputs

#lists to store accuaryc of each fold
accuracy_tree50 = []
accuracy_tree500 = []
accuracy_tree1000 = []
accuracy_forrest50 = []
accuracy_forrest500 = []
accuracy_forrest10000 = []

#the desgined classifiers
tree50 = neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'sgd',
                                  max_iter = 10000, hidden_layer_sizes = (50, 50),
                                  activation = 'logistic',
                                  verbose = False, tol = 1e-10,
                                  random_state = 11)

tree500 = neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'sgd',
                                  max_iter = 10000, hidden_layer_sizes = (500, 500),
                                  activation = 'logistic',
                                  verbose = False, tol = 1e-10,
                                  random_state = 11)

tree1000 = neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'sgd',
                                  max_iter = 10000, hidden_layer_sizes = (1000, 1000),
                                  activation = 'logistic',
                                  verbose = False, tol = 1e-10,
                                  random_state = 11)

forrest50 = ensemble.RandomForestClassifier(n_estimators=50, min_samples_leaf=10)
forrest500 = ensemble.RandomForestClassifier(n_estimators=500, min_samples_leaf=10)
forrest10000 = ensemble.RandomForestClassifier(n_estimators=10000, min_samples_leaf=10)

#number of folds
split=10

kf = KFold(n_splits=split, random_state=None, shuffle=True)#sets the folds and shuffles.

#for each fold
for train_index, test_index in kf.split(x):
    print("TRAIN: ", train_index, "TEST:", test_index)
    X_train, X_test = x.iloc[train_index,:],x.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index],y.iloc[test_index,:]
    #splits tehe suffled data

    #trains the models and predicts usiing the test data for each fold
    tree50.fit(X_train,y_train)
    predict_tree50 = tree50.predict(X_test)
    accuracytree50 = accuracy_score(predict_tree50,y_test)
    accuracy_tree50.append(accuracytree50)#checks the accuracy of each fold and appends to find the mean later

    tree500 = tree500.fit(X_train,np.ravel(y_train))
    predict_tree500 = tree500.predict(X_test)
    accuracytree500 = accuracy_score(predict_tree500,y_test)
    accuracy_tree500.append(accuracytree500)

    tree1000 = tree1000.fit(X_train,np.ravel(y_train))
    predict_tree1000 = tree1000.predict(X_test)
    accuracytree1000 = accuracy_score(predict_tree1000,y_test)
    accuracy_tree1000.append(accuracytree1000)

    forrest50 = forrest50.fit(X_train,np.ravel(y_train))
    predict_forrest50 = forrest50.predict(X_test)
    accuracyforrest50 = accuracy_score(predict_forrest50,y_test)
    accuracy_forrest50.append(accuracyforrest50)

    forrest500 = forrest500.fit(X_train,np.ravel(y_train))
    predict_forrest500 = forrest500.predict(X_test)
    accuracyforrest500 = accuracy_score(predict_forrest500,y_test)
    accuracy_forrest500.append(accuracyforrest500)

    forrest10000 = forrest10000.fit(X_train,np.ravel(y_train))
    predict_forrest10000 = forrest10000.predict(X_test)
    accuracyforrest10000 = accuracy_score(predict_forrest10000,y_test)
    accuracy_forrest10000.append(accuracyforrest10000)


accuracy_tree50_result = sum(accuracy_tree50)/split #finds the mean accuracy
accuracy_tree500_result = sum(accuracy_tree500)/split
accuracy_tree1000_result = sum(accuracy_tree1000)/split
accuracy_forrest50_result = sum(accuracy_forrest50)/split
accuracy_forrest500_result = sum(accuracy_forrest500)/split
accuracy_forrest10000_result = sum(accuracy_forrest10000)/split

print("Accuracy of ANN with 50 neurons:")
print(accuracy_tree50)
print(accuracy_tree50_result)

print("Accuracy of ANN with 500 neurons:")
print(accuracy_tree500)
print(accuracy_tree500_result)

print("Accuracy of ANN with 1000 neurons:")
print(accuracy_tree1000)
print(accuracy_tree1000_result)

print("Accuracy of Forrest with 50 trees:")
print(accuracy_forrest50)
print(accuracy_forrest50_result)

print("Accuracy of Forrest with 500 trees:")
print(accuracy_forrest500)
print(accuracy_forrest500_result)

print("Accuracy of Forrest with 10000 trees:")
print(accuracy_forrest10000)
print(accuracy_forrest10000_result)