import statsmodels.discrete.discrete_model as sm
import random
import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
np.random.seed(2020)
random.seed(2020)

class BestSubsetSelectionOLS(BaseEstimator):
    
    def __init__(self, fK=3):
        self.fK = fK
    
    def myBic(self, n, mse, k):
        if k<=0:
            return np.nan
        else:
            return n*np.log(mse) + k*np.log(n)
    
    def processSubset(self, X, y, feature_set):
        regr = sm.Logit(y, sm.tools.add_constant(X[list(feature_set)])).fit(disp=0)
        return{"model": regr, "bic": regr.bic, "best_predictors": feature_set}
    
    
    def getBest(self, X, y, fK):
        results=[]
        X=pd.DataFrame(X)
        for combo in itertools.combinations(X.columns, fK):
            results.append(self.processSubset(X,y,combo))
        models = pd.DataFrame(results)
        best_model = models.loc[models["bic"].idxmin(), 'model']
        best_predictors = models.loc[models["bic"].idxmin(), 'best_predictors']
        return best_model, best_predictors
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.best_model, self.best_predictors = self.getBest(X, y, self.fK)
        self.is_fitted = True
        return self
    
    def get_params(self, deep=True):
        return {"fK": self.fK}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
   
