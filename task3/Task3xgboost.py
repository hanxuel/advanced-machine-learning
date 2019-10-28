# xgboost

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score
from scipy import stats
from minepy import MINE
from numpy import array
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.neural_network import MLPClassifier


def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

def merge(df1,df2):
    df1=np.array(df1)
    df2=np.array(df2)
    df=np.hstack((df1,df2))
    return df

# Removing columns/rows
def remove(X1):
    X = np.delete(X1.values, 0, axis=1)
    X = np.delete(X, 46, axis=1)
    return X



# def score(y_predict,y_test):
#     y_mean=np.mean(y_test)
#     score=1-np.sum((y_test-y_predict)*(y_test-y_predict))/np.sum((y_test-y_mean)*(y_test-y_mean))
#     return score

# read data
dfX_train = pd.read_csv('Xtrainfeature.csv')
dfy_train = pd.read_csv('y_train.csv')
dfX_test = pd.read_csv('Xtestfeature.csv')
#dfX_train1=pd.read_csv('X_trainfeature1.csv')
#dfX_test1=pd.read_csv('X_testfeature1.csv')


#dfX_train=merge(dfX_train,dfX_train1)
dfX_train=pd.DataFrame(dfX_train)
#dfX_test=merge(dfX_test,dfX_test1)
dfX_test=pd.DataFrame(dfX_test)

# Missing Value Processing
X_train=dfX_train.fillna(dfX_train.mean())
X_test=dfX_test.fillna(dfX_test.mean())
# print (X_train.shape)
# print (dfy_train.shape)

# Removing all zero column
#X_train = X_train.iloc[:,(X_train.values != 0).any(axis=0)]
#X_train = X_train[~(X_train==0).all(1)]
#X_test = X_test.iloc[:,(X_train.values != 0).any(axis=0)]

# Removing columns/rows
# X_train = np.delete(X_train.values,0,axis=1)
# X_train=np.delete(X_train, 46, axis=1)
X_train=remove(X_train)
X_test=remove(X_test)

y_train = np.delete(dfy_train.values,0,axis=1)
# print(X_train.shape)
# y_train=y_train[0:-1]
# print(X_test.shape)

# K-fold cross validation
X,y = shuffle(X_train,y_train)
# kf = KFold(n_splits=10)
# i=0
# F1=np.zeros(10)
# max_score = 0
# y_tpred = []
#
# for train_index, test_index in kf.split(X):
#
#     X_train, X_val = X[train_index], X[test_index]
#     y_train, y_val = y[train_index], y[test_index]
#
#     #undersampling
#     # rus = RandomUnderSampler(random_state=0)
#     # X_sample, y_sample = rus.fit_sample(X_train, y_train)
#
#     X_sample=X_train
#     y_sample = y_train
#
#     # Standardlization
#     scaler = StandardScaler()
#     scaler.fit(X_sample)
#     Xn_train = scaler.transform(X_sample)
#     Xn_val = scaler.transform(X_val)
#     Xn_test = scaler.transform(X_test)
#     y_train = y_sample.flatten()
#
#     #Univariate feature selection
#     selector =SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:mic(x, Y), X.T))).T)), k=100)
#     # selector = SelectKBest(lambda X, Y: np.array(map(lambda x: np.abs(pearsonr(x, Y)), X.T))[:, 0].T, k=500)
#     selector.fit(Xn_train, y_train)
#     Xn_train = selector.transform(Xn_train)
#     Xn_val = selector.transform(Xn_val)
#     Xn_test = selector.transform(Xn_test)
#
#     # xgboosting model
#     #clf = SVC(gamma='auto')
#
#     # clf=SVC(C=1, kernel='rbf', class_weight='balanced')
#
#     clf = xgb.XGBClassifier(n_estimators=450, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=6)
#     clf.fit(Xn_train,y_train)
#     # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
#     # clf.fit(Xn_train, y_train)
#     y_predict = clf.predict(Xn_val)
#     F1[i] = f1_score(y_val, y_predict, average='weighted')
#     print (F1[i])
#
#     max_y_result = clf.predict(Xn_test)
#     a = np.array(range(0, max_y_result.shape[0]), dtype=np.float64)
#     b = np.array(max_y_result, dtype=np.float64)
#     result = pd.DataFrame(columns=['id', 'y'], data=np.array([a, b]).T)
#     result.to_csv('AIchemist%i.csv' % i)
#     i=i+1

scaler = StandardScaler()
scaler.fit(X)
Xn_train = scaler.transform(X)
Xn_test = scaler.transform(X_test)
selector =SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:mic(x, Y), X.T))).T)), k=100)
print (Xn_train.shape,y.shape)
y=y.flatten()
selector.fit(Xn_train, y)
Xn_train = selector.transform(Xn_train)
Xn_test = selector.transform(Xn_test)
clf = xgb.XGBClassifier(n_estimators=400, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=6)
# clf=SVC(C=1, kernel='rbf', class_weight='balanced')
clf.fit(Xn_train,y)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
# clf.fit(Xn_train, y)
y_predict = clf.predict(Xn_test)
a = np.array(range(0, y_predict.shape[0]), dtype=np.float64)
b = np.array(y_predict, dtype=np.float64)
result = pd.DataFrame(columns=['id', 'y'], data=np.array([a, b]).T)
result.to_csv('result.csv')
# print(np.mean(F1))


