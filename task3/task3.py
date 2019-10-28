import pandas as pd
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
from sklearn.decomposition import PCA
from collections import Counter
from pandas import DataFrame
from imblearn.combine import SMOTEENN
from sklearn.metrics import f1_score


# def score(y_predict,y_test):
#     y_mean=np.mean(y_test)
#     score=1-np.sum((y_test-y_predict)*(y_test-y_predict))/np.sum((y_test-y_mean)*(y_test-y_mean))
#     return score

# read data
dfX_train = pd.read_csv('Xtrainfeature.csv')
print(dfX_train.shape)
dfy_train = pd.read_csv('y_train.csv')
dfX_test = pd.read_csv('Xtestfeature.csv')
print(dfX_test.shape)
# Missing Value Processing
X_train=dfX_train.fillna(dfX_train.mean())
X_test=dfX_test.fillna(dfX_test.mean())

# Removing all zero column
X_train = X_train.iloc[:,(X_train.values != 0).any(axis=0)]
X_train = X_train[~(X_train==0).all(1)]
X_test = X_test.iloc[:,(X_train.values != 0).any(axis=0)]

# Removing columns/rows
X_train = np.delete(X_train.values,0,axis=1)
y_train = np.delete(dfy_train.values,0,axis=1)
X_test = np.delete(X_test.values,0,axis=1)

print(y_train.shape)

# K-fold cross validation
X,y = shuffle(X_train,y_train)
kf = KFold(n_splits=10)
i=0
# BMAC=np.zeros(10)
max_score = 0
y_tpred = []

for train_index, test_index in kf.split(X):

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    #undersampling
    rus = RandomUnderSampler(random_state=0)
    X_sample, y_sample = rus.fit_resample(X_train, y_train)

    # smote_enn = SMOTEENN(random_state=0)
    # X_sample, y_sample = smote_enn.fit_sample(X_train, y_train)
    print (y_sample.shape)
    print (np.bincount(np.ravel(y_sample.astype(np.int32))))


    # Standardlization
    scaler = StandardScaler()
    scaler.fit(X_sample)
    Xn_train = scaler.transform(X_sample)
    Xn_val = scaler.transform(X_val)
    Xn_test = scaler.transform(X_test)

    y_train = y_sample.flatten()

    #Univariate feature selection
    # selector = SelectKBest(lambda X, Y: np.array(map(lambda x: np.abs(pearsonr(x, Y)), X.T))[:, 0].T, k=500)
    # selector.fit(Xn_train, y_train)
    # Xn_train = selector.transform(Xn_train)
    # Xn_val = selector.transform(Xn_val)
    # Xn_test = selector.transform(Xn_test)

    # pca=PCA(n_components=200)
    # pca.fit(Xn_train)
    # Xn_train1=pca.transform(Xn_train)
    # Xn_val1=pca.transform(Xn_val)
    # Xn_test1=pca.transform(Xn_test)



    # xgboosting model
    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=6)
    clf.fit(Xn_train,y_train)
    y_predict = clf.predict(Xn_val)
    F1 = f1_score(y_val, y_predict, average='micro')
    print('F1:', F1)

    max_y_result = clf.predict(Xn_test)
    a = np.array(range(0, max_y_result.shape[0]), dtype=np.float64)
    b = np.array(max_y_result, dtype=np.float64)
    result = pd.DataFrame(columns=['id', 'y'], data=np.array([a, b]).T)
    result.to_csv('AIchemist%i.csv' % i)
    # if BMAC[i]>=0.69:
    #     max_y_result=clf.predict(Xn_test)
    i=i+1