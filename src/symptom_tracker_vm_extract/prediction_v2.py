import statsmodels.discrete.discrete_model as sm
import stepwise_selector
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score
from imv_functions import ll, get_w, minimize_me, get_ew
import numpy as np
from tqdm.notebook import tqdm


def get_ys(X, Y, train_index, test_index, feature_select):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    if feature_select==False:
        regr = sm.Logit(Y_train, X_train).fit(disp=0)
    else:
        regr = stepwise_selector.BestSubsetSelectionOLS(fK=5)
        regr.fit(X_train, Y_train)
        regr = regr.best_model
        X_test = sm.tools.add_constant(X_test.iloc[:,regr.model.exog_names[1:]])
    Y_pred = regr.predict(X_test)
    return Y_train, Y_test, Y_pred


def predict_oos_compare(df1, df2, nfolds, feature_select, states, stratified=False):
    X1 = sm.tools.add_constant(df1.drop('y', 1))
    Y1 = df1['y']
    X2 = sm.tools.add_constant(df2.drop('y', 1))
    Y2 = df2['y']
    imv=[]
    kf=StratifiedKFold(n_splits=nfolds)
    kf.get_n_splits(X1, Y1)
    for train_index, test_index in kf.split(X1, Y1):
        Y1_train, Y1_test, Y1_pred = get_ys(X1, Y1, train_index,
                                            test_index, feature_select)
        Y2_train, Y2_test, Y2_pred = get_ys(X2, Y2, train_index,
                                            test_index, feature_select)
        imv.append(get_ew(get_w(ll(Y1_test, Y1_pred)),
                          get_w(ll(Y1_test, Y2_pred))))
    return imv



def predict_oos(df, nfolds, feature_select, states, stratified=False):
    X = sm.tools.add_constant(df.drop('y', 1))
    Y = df['y']
    roc_auc=[]
    imv=[]
    r2=[]
    if states is None:
        if stratified == False: #can be turned into a function
            kf=KFold(n_splits=nfolds)
            kf.get_n_splits(X)
            for train_index, test_index in kf.split(X):
                try:
                    Y_train, Y_test, Y_pred = get_ys(X, Y, train_index,
                                                     test_index, feature_select)
                    roc_auc.append(roc_auc_score(Y_test,Y_pred))
                    r2.append(r2_score(Y_test, Y_pred))
                    imv.append(get_ew(get_w(ll(Y_test, np.mean(Y_train))),
                                           get_w(ll(Y_test, Y_pred))))
                except:
                    roc_auc.append(np.nan)
                    r2.append(np.nan)
                    imv.append(np.nan)
        else:
            kf=StratifiedKFold(n_splits=nfolds)
            kf.get_n_splits(X, Y)
            for train_index, test_index in kf.split(X, Y):
                try:
                    Y_train, Y_test, Y_pred = get_ys(X, Y, train_index,
                                                     test_index, feature_select)
                    r2.append(r2_score(Y_test, Y_pred))
                    roc_auc.append(roc_auc_score(Y_test,Y_pred))
                    imv.append(get_ew(get_w(ll(Y_test, np.mean(Y_train))),
                                           get_w(ll(Y_test, Y_pred))))
                except:
                    r2.append(np.nan)
                    roc_auc.append(np.nan)
                    imv.append(np.nan)
    else:
        for state in tqdm(states):
            imv_state = []
            roc_auc_state = []
            r2_state = []
            if stratified == False:
                kf=KFold(n_splits=nfolds, shuffle=True, random_state=state)
                kf.get_n_splits(X)
                for train_index, test_index in kf.split(X):
                    try:
                        Y_train, Y_test, Y_pred = get_ys(X, Y, train_index,
                                                         test_index, feature_select)
                        r2_state.append(r2_score(Y_test, Y_pred))
                        roc_auc_state.append(roc_auc_score(Y_test,Y_pred))
                        imv_state.append(get_ew(get_w(ll(Y_test, np.mean(Y_train))), get_w(ll(Y_test, Y_pred))))
                    except:
                        r2_state.append
                        roc_auc_state.append(np.nan)
                        imv_state.append(np.nan)
            else:
                kf=StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                kf.get_n_splits(X, Y)
                for train_index, test_index in kf.split(X, Y):
                    try:
                        Y_train, Y_test, Y_pred = get_ys(X, Y, train_index,
                                                         test_index, feature_select)
                        r2_state.append(r2_score(Y_test, Y_pred))
                        roc_auc_state.append(roc_auc_score(Y_test,Y_pred))
                        imv_state.append(get_ew(get_w(ll(Y_test, np.mean(Y_train))), get_w(ll(Y_test, Y_pred))))
                    except:
                        r2_state.append
                        roc_auc_state.append(np.nan)
                        imv_state.append(np.nan)
            r2.append(r2_state)
            roc_auc.append(roc_auc_state)
            imv.append(imv_state)
    return r2, roc_auc, imv


def print_results(roc_auc_list, imv_list, axis=0):
    if axis==0:
        print("AUC-ROC mean, min, max, sd:",
              np.mean(roc_auc_list),
              np.min(roc_auc_list),
              np.max(roc_auc_list),
              np.std(roc_auc_list))
        print("IMV mean, min, max, sd:",
              np.mean(imv_list),
              np.min(imv_list),
              np.max(imv_list),
              np.std(imv_list))
    elif axis==1:
        print("AUC-ROC mean, min, max, sd:",
              np.mean(roc_auc_list, axis=axis),
              np.min(roc_auc_list, axis=axis),
              np.max(roc_auc_list, axis=axis),
              np.std(roc_auc_list, axis=axis))
        print("IMV mean, min, max, sd:",
              np.mean(imv_list, axis=axis),
              np.min(imv_list, axis=axis),
              np.max(imv_list, axis=axis),
              np.std(imv_list, axis=axis))