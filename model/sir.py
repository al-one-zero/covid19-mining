import numpy as np
from pandas import DataFrame
from .base import BaseModel

class SIRModel(BaseModel):
    def __init__(self, N=None, beta:float=0.1, gamma:float=0.2):
        self.beta_ = beta
        self.gamma_ = gamma
        self.N_ = N

    @property
    def params(self):
        return self.beta_,self.gamma_

    @params.setter
    def params(self,new_params):
        self.beta_,self.gamma_=new_params

    def _update_params(self,new_params):
        self.beta_,self.gamma_=new_params

    def deriv(self, t, y, beta, gamma):
        N = self.N_ = sum(y)
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def _check_input(self, y):
        if isinstance(y, list):
            S, I, R = y
            I0=I[0]
            S0 = S
            R0 = R
            y_0=(S0,I0,R0)
        elif isinstance(y, DataFrame):
            y=y.values
            y_0 = y.iloc[0].values
        elif isinstance(y, np.ndarray):
            S, I, R = y
            y_0=(S[0],I[0],R[0])
        else:
            raise ValueError
        return y, y_0

    def fit(self, t, y, N=None):
        if N is None:
            self.N_ = N
        y, y_0 = self._check_input(y)
        y_s = np.hstack(y[1])
        def f(t,*params):
            y_t=self._predict(t,y_0,params)
            pred=np.hstack(y_t[1])
            return pred
        self._curve_fit(f,t,y_s)
        return self
