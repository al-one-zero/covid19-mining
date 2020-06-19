import numpy as np
from pandas import DataFrame
from . import SEIRModel as SEIR

class SEIRDModel(SEIR):

    def __init__(self, rho=0.333, alpha=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_ = rho
        self.alpha_ = alpha
        self.param_names.extend(('ρ', 'α'))

    @property
    def params(self):
        return super().params + (self.rho_, self.alpha_)

    @params.setter
    def params(self, new_params):
        self._update_params(*new_params)
    
    def _update_params(self, args):
        print(args)
        super()._update_params(args[:-2])
        self.rho_, self.alpha_ = args[-2:]

    def deriv(self, t, y, beta, gamma, delta, rho, alpha):
        self.N_ = N = sum(y)
        S, E, I, R, D = y

        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
        dRdt = (1 - alpha) * gamma * I
        dDdt = alpha * rho * I

        return dSdt, dEdt, dIdt, dRdt, dDdt

    def _check_input(self, y):
        if isinstance(y, list):
            S, E, I, R, D = y
            I0=I[0]
            E0 = E
            S0 = S
            R0 = R
            D0 = D[0]
            y_0=(S0, E0, I0, R0, D0)
        elif isinstance(y, DataFrame):
            y=y.values
            y_0 = y.iloc[0].values
        elif isinstance(y, np.ndarray):
            S, E, I, R, D = y
            y_0=(S[0],E[0],I[0],R[0],D[0])
        else:
            raise ValueError
        return y, y_0


    def fit(self, t, y, N=None):
        if N is not None:
            self.N_ = N
        y, y_0 = self._check_input(y)
        y_s = np.hstack(y[2::2])
        def f(t,*params):
            y_t=self._predict(t,y_0,params)
            pred=np.hstack(y_t[2::2])
            return pred
        self._curve_fit(f,t,y_s)
        return self
