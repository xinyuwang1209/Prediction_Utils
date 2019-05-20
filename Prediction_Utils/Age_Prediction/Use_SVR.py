__auther__ = 'Xinyu Wang'

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
def run_CrossValidation(X,y,cv=5,return_format='np'):
    # Memo
    clf = svm.SVC(kernel='linear', C=1)
    clf = svm.SVC(kernel='rbf', C=1)
    clf = svm.SVC(kernel='poly’', C=1)
    clf = svm.SVC(kernel='precomputed', C=1)
    clf = svm.SVC(kernel='sigmoid', C=1)
    # scores = cross_validate(clf, X, y, cv=cv)

# linear
# rbf                   gamma
# poly          degree  gamma   coef0
# precomputed
# sigmoid               gamma   coef0

def run_svm_cv(X,y,cv=5,kernel='linear',C=1,degree=1,tol=0.001,gamma='scale',coef0=1):
    clf = svm.SVC(kernel=kernel, C=C,degree=degree,tol=tol,gamma=gamma,coef0=coef0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(X,y)
    if kernel == 'linear':
        return clf.coef_
    else:
        return clf.dual_coef_

kernels = ['linear','rbf','poly','precomputed','sigmoid']

# def run_svr(X,y)
#
#     clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
#     clf.fit(X, y)
#
#     # Read out attributes
#     coeffs = lasso.coef_         # dense np.array
#     # coeffs = lasso.sparse_coef_  # sparse matrix
#
#     # coeffs = lasso.intercept_    # probably also relevant
#     if return_format == 'np':
#         return coeffs
#     else:
#         # TODO
#         return coeffs
