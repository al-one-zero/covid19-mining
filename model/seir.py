import numpy as np
from pandas import DataFrame
from . import SIRModel as SIR

class SEIRModel(SIR):

    def __init__(self, delta=0.333, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_ = delta

    @property
    def params(self):
        return *super().params, self.delta_

    @params.setter
    def params(self, new_params):
        self._update_params(*new_params)

    def _update_params(self, args):
        self.beta_, self.gamma_, self.delta_ = args

    def deriv(self, t, y, beta, gamma, delta):
        self.N_ = N = sum(y)
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = delta * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def _check_input(self, y):
        if isinstance(y, list):
            S, E, I, R = y
            I0=I[0]
            E0 = E[0]
            S0 = S
            R0 = R
            y_0=(S0, E0, I0, R0)
        elif isinstance(y, DataFrame):
            y=y.values
            y_0 = y.iloc[0].values
        elif isinstance(y, np.ndarray):
            S, E, I, R = y
            y_0=(S[0],E[0],I[0],R[0])
        else:
            raise ValueError
        return y, y_0


    def fit(self, t, y, N=None):
        if N is None:
            self.N_ = N
        y, y_0 = self._check_input(y)
        y_s = np.hstack(y)
        def f(t,*params):
            y_t=self._predict(t,y_0,params)
            pred=np.hstack(y_t)
            return pred
        self._curve_fit(f,t,y_s)
        return self
