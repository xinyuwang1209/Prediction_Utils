import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import KFold
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
import pathos
import os
import pickle

data=pd.read_csv("data_unqf.csv")
Gender=pd.read_csv("HCP_summary_S1206.csv")
Gender=Gender[["Subject","Gender"]]
Age=pd.read_csv("psychiatric_data_HCP.csv")
Age=Age[["Subject","Age_in_Yrs"]]
Filter=pd.read_csv("x_unqf.csv")

data=data.merge(Gender,how="inner")
data=data.merge(Age,how="inner")
#data=data[data["Gender"]=="F"]
Age=data["Age_in_Yrs"]
Gender=data["Gender"].map({"M":0,"F":1})

y=data["addiction"]
data=data.drop(["addiction","Gender","Age_in_Yrs"],axis=1)
#data=data[Filter.columns]
# data=pd.concat([data.mul(Gender,axis=0).mul(Age,axis=0),
#                 data.mul(Gender,axis=0),
#                 data.mul(Age,axis=0),
#                 data,
#                 Gender,
#                 Age], axis=1, sort=False)

# data=pd.concat([data,
#                 data.mul(Gender,axis=0),
#                 Gender], axis=1, sort=False)
data=pd.concat([data], axis=1, sort=False)


print(data.shape)
X=data.iloc[:,1:]
# idea: train and collect test_score and cv_score
# find best with C

params = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e0,5e0,1e1,5e1,1e2]


def evaluate_LogisticRegression(X,y,dir,name='LogisticRegression',
                                params = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e0,5e0,1e1,5e1,1e2],
                                test=False):
    if dir[-1] is not '/':
        dir = dir + '/'
    if test:
        params = params[:2]
    len_params = len(params)
    param_result = pd.DataFrame(columns=['param','best_random_state','avg_train_score','avg_test_score','avg_cv_score','best_plt'])
    directory = dir + name
    if not os.path.exists(directory):
            os.makedirs(directory)
    for i in range(len_params):
        param = params[i]
        clf = LogisticRegression(solver='liblinear',penalty='l2',C=param)
        if test:
            pd_result = evaluate_clf(X,y,clf,random_states=[42,50])
        else:
            pd_result = evaluate_clf(X,y,clf)
        # save result
        pickle.dump(pd_result,open(directory + '/LogisticRegression' + str(param),'wb'))
        # get avg_train_score,
        current = {'param': param,
                   'best_random_state': pd_result.loc[pd_result['test_score'] == pd_result['test_score'].max()].iloc[0,:]['random_state'],
                   'avg_train_score': pd_result['train_score'].mean(),
                   'avg_test_score': pd_result['test_score'].mean(),
                   'avg_cv_score': pd_result['cv_score'].mean(),
                   'best_plt': pd_result.loc[pd_result['test_score'] == pd_result['test_score'].max()].iloc[0,:]['plt']}
        param_result.loc[i] = current
    pickle.dump(param_result,open(directory + '/LogisticRegression', 'wb'))
    return param_result



def evaluate_clf(X,y,clf,mp=False,test_size=0.25,random_states=[0,10,20,30,40,42,50]):
    total_start_time = time.time()
    # Create result
    rs_result = pd.DataFrame(columns=['random_state','train_score','test_rate','test_score','cv_result','cv_score','plt','elapsed_time'])
    len_random_states = len(random_states)
    best_cv_score = 0
    best_test_score = 0
    if mp:
        pd.DataFrame(X).to_csv('/dev/shm/HCP_X.csv',index=False)
        pd.DataFrame(y).to_csv('/dev/shm/HCP_y.csv',index=False)
        ncpu = cpu_count()
        chunk_size = math.ceil(len_random_states/ncpu)
        pool = pathos.multiprocessing.ProcessingPool(ncpu-1).map
        result = pool(evaluate_clf_one,random_states,chunksize=chunk_size)
        rs_result = pd.DataFrame(result)
        os.remove('/dev/shm/HCP_X.csv')
        os.remove('/dev/shm/HCP_y.csv')
    else:
        for i in range(len_random_states):
            start_time = time.time()
            random_state = random_states[i]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            clf.fit(X_train, y_train)
            result = clf.predict_proba(X_test)
            train_score = clf.score(X_train, y_train)
            test_rate = sum(y_test==0)/len(y_test)
            test_score = clf.score(X_test, y_test)
            cv_result = cross_val_score(clf, X, y, cv=KFold(4, shuffle=True, random_state=0))
            cv_score = cv_result.mean()
            fpr, tpr, thresholds = metrics.roc_curve( y_test,result[:,1])
            myplt = plt.figure()
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='result')
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            elapsed_time = int(time.time() - start_time)
            rs_result.loc[i] = [random_state,train_score,test_rate,test_score,cv_result,cv_score,myplt,elapsed_time]
            if best_cv_score < cv_score:
                best_cv_score = cv_score
            if best_test_score < test_score:
                best_test_score = test_score
            print(i+1,'/',len_random_states,"Elapsed Time:",elapsed_time)
            print("Best Test Score:", best_test_score)
            print("Best CV Score", best_cv_score)
    print("Total Elapsed_time:", int(time.time()-total_start_time))
    return rs_result


def evaluate_clf_one(random_state):
    start_time = time.time()
    X = pd.read_csv('/dev/shm/HCP_X.csv')
    y = pd.read_csv('/dev/shm/HCP_y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    clf.fit(X_train, y_train)
    result = clf.predict_proba(X_test)
    train_score = clf.score(X_train, y_train)
    test_rate = sum(y_test==0)/len(y_test)
    test_score = clf.score(X_test, y_test)
    cv_result = cross_val_score(clf, X, y, cv=KFold(4, shuffle=True, random_state=0))
    cv_score = cv_result.mean()
    fpr, tpr, thresholds = metrics.roc_curve( y_test,result[:,1])
    myplt = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='result')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    elapsed_time = int(time.time() - start_time)
    return {'random_state': random_state,
            'train_score':  train_score,
            'test_rate':    test_rate,
            'test_score':   test_score,
            'cv_result':    cv_result,
            'cv_score':     cv_score,
            'plt':          myplt,
            'elapsed_time': elapsed_time}

# Usage
clf = LogisticRegression(solver='liblinear',penalty='l2',C=5)
rs_result = evaluate_clf(X,y,clf,mp=False)
rs_result[['test_rate','test_score','cv_score']]



plt = test(clf)
print(clf.score(X_train, y_train))
plt.show()


def fig_barh(ylabels, xvalues, title=''):
    # create a new figure
    fig = plt.figure()
    # plot to it
    yvalues = 0.1 + np.arange(len(ylabels))
    plt.barh(yvalues, xvalues, figure=fig)
    yvalues += 0.4
    plt.yticks(yvalues, ylabels, figure=fig)
    if title:
        plt.title(title, figure=fig)
    # return it
    return fig

# Usage
param_result = evaluate_LogisticRegression(X,y,os.getcwd(),'LogisticRegression975_6000',test=False)
pickle.dump(param_result,open(os.getcwd() + '/LogisticRegression975_6000/LogisticRegression','wb'))
best_plt = param_result.loc[param_result['avg_cv_score'] == param_result['avg_cv_score'].max()].iloc[0,:]['best_plt']


def result extraction
