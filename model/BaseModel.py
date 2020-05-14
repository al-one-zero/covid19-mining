import numpy as np 
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

class BaseModel():
    def __init__(self):
        self.params=None
        self.params_cov_=None

    def deriv(self,*args):
        raise NotImplemented

    def _update_params(self,new_params):
        self.params=self.__class__.Parameter(*new_params)

    def _curve_fit(self,f,t,y,bounds=(0,np.inf)):
        params_init=self.params
        fitted_params,pcov=curve_fit(f,t,y,p0=params_init,bounds=bounds)
        self.params_cov_=pcov
        self._update_params(fitted_params)

    def _integrate(self,t,y_0,params):
        t_span=(min(t),max(t))
        deriv=lambda t,y: self.deriv(t,*y,*params)
        ode_result=solve_ivp(deriv,t_span,y_0,t_eval=t)
        return ode_result

    def _predict(self,t,y_0,params):
        ode_result=self._integrate(t,y_0,params)
        y_t=ode_result.y
        return y_t

    def predict(self,t,y_0):
        return self._predict(t,y_0,self.params)