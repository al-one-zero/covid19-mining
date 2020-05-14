import numpy as np
from BaseModel import BaseModel

class SIRModel(BaseModel):
    def __init__(self, beta:float=0.1, gamma:float=0.2):
        self.beta_ = beta
        self.gamma_ = gamma

    @property
    def params(self):
        return self.beta_,self.gamma_

    @params.setter
    def params(self,new_params):
        self.beta_,self.gamma_=new_params

    def _update_params(self,new_params):
        self.beta_,self.gamma_=new_params
    
    def deriv(self,t,S,I,R,beta,gamma):
        N=S+I+R
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def fit(self,t, N,I,R):
        y=np.hstack((I,R))
        I0=I[0]
        R0=R[0]
        S0=N-I0-R0
        y_0=(S0,I0,R0)
        def f(t,*params):
            y_t=self._predict(t,y_0,params)
            I_pred=y_t[1]
            R_pred=y_t[2]
            pred=np.hstack((I_pred,R_pred))
            return pred
        self._curve_fit(f,t,y)
        return self



