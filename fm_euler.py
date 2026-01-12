import torch
from torchdyn.numerics.solvers.ode import Euler

class FmEuler(Euler):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)
    
    def step(self, f, x, t, dt, k1=None, args=None):
        t = torch.ones(x.shape[0]).to(x.device) * t
        if args['cond']:
            cond = args['label']
        else:
            cond = None
        if k1 == None: k1 = f(t, x, cond)
        x_sol = x + dt * k1
        return None, x_sol, None

'''
    def step(self, f, x, t, dt, k1=None, args=None):
        if k1 == None: k1 = f(t, x)
        x_sol = x + dt * k1
        return None, x_sol, None
'''
