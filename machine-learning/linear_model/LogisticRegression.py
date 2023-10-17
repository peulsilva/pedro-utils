import numpy as np
import pandas as pd
from scipy.stats import norm

class LogisticRegression():
    def __init__(self) -> None:
        self.coef_ = None
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def fit(
        self,
        X : pd.DataFrame,
        y : pd.DataFrame,
        err : float = 1e-5
    ):
        columns = X.columns
        y = y.to_numpy()
        X = X.to_numpy()

        n = len(X)
        p = X.shape[1]


        self.coef_ = np.zeros([1, p])
        W = np.eye(n)
        for i in range(n):
            W[i, i] = self.d_sigmoid(self.coef_ @ X[i])


        phi_theta = np.array([self.sigmoid(self.coef_ @ X[i]) for i in range(n)])

        while np.linalg.norm(X.T @ (y - phi_theta)) > err:
            
            z = X @ self.coef_.T + np.linalg.inv(W) @ (y - phi_theta)
            self.coef_ = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ z
            
            self.coef_ = self.coef_.T

            W = np.eye(n)
            for i in range(n):
                W[i, i] = self.d_sigmoid(self.coef_ @ X[i])
            
            phi_theta = np.array([self.sigmoid(self.coef_ @ X[i]) for i in range(n)])

        coef = pd.Series(
            self.coef_.squeeze(),
            index = columns
        )

        self.coef_ = coef
        self.std = self.__std(X, columns)

    def predict_proba(self, X : pd.DataFrame):
        X = X.reindex_like(self.coef_)\
            .to_numpy()\
            .reshape(1,-1)
        
        probas = np.array([self.sigmoid(self.coef_.T @ x_i.T ) for x_i in X])        
        
        return probas

    def __std(self, X : pd.DataFrame, cols):

        n, p = X.shape

        W = np.eye(n)
        for i in range(n):
            W[i, i] = self.d_sigmoid(self.coef_ @ X[i])

        std =  np.sqrt(np.linalg.inv(X.T @ W @ X))

        res = {}

        for i in range(p):
            res[cols[i]] = std[i,i]

        return pd.Series(res)
    
    def confidence_interval(self, alpha):
        high = self.coef_ + norm.ppf(1 - alpha / 2) * self.std
        low = self.coef_ + norm.ppf(alpha/2) * self.std

        conf_int_df =  pd.concat([low, high], axis = 1)

        conf_int_df.rename(
            columns = {
                0 : "lower bound",
                1: "upper bound"
            },
            inplace= True
        )

        return conf_int_df