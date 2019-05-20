__auther__ = 'Xinyu Wang'

from sklearn.linear_model import Lasso, LassoCV
import sklearn.model_selection.train_test_split
from sklearn.model_selection import cross_val_score

def run_lasso_cv(X,y,cv=5,random_state=0,C=1,degree=1,tol=0.001):
    clf = svm.SVC(kernel='linear', C=C,degree=degree,tol=tol)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(X,y)
    return clf.coef_

#


# def feature_entry_locator(n):
#     current = n
#     n_column = 115
#     row = 0
#     while n_column <= current:
#         current -= n_column
#         n_column -= 1
#         row += 1
#     column = row + current + 1
#     return row, column

# Get best 100 entries
features_index = [i[0] for i in sorted(enumerate(clf.coef_[0]), key=lambda x:x[1])][:10]
best_10_features = data.iloc[:,[0]+[i+1 for i in features_index]]
best_10_features.to_csv('/shared/healthinfolab/hcpdata/aal_corr_matrices/best_10_features.csv')



    # run 10 times , get linear model, average the coefficient
    # bagging boostraping aggregating
    # ddiction 5 abus
    # {value for  in variable}
    # take a look at fslnet

    # 1 for all correlation matrix
    # 2 for two column
    # 3

    # april june oct
    # all phenotypr in mental disease are questionnares
