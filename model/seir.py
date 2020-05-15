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
        self.beta_, self.gamma_, self.delta_ = new_params

    def _update_params(self, *args):
        self.params = args

    def deriv(self, t, y, beta, gamma, delta):
        N = sum(y)
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def fit(self, t, N, I, R):
        self.N_ = N
        y=np.hstack((I))
        I0=I[0]
        S0=N-I0
        y_0=(S0,I0,0)
        def f(t,*params):
            y_t=self._predict(t,y_0,params)
            I_pred=y_t[1]
            pred=np.hstack((I_pred))
            return pred
        self._curve_fit(f,t,y)
        return self
