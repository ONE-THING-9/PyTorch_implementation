from library import *

class SGD:
  def __init__(self, params , lr , wd = 0):
    params = list(params)
    fc.store_attr()
    self.i = 0

  def step(self):
    with torch.no_grad():
      for p in self.params:
        self.reg_step(p)
        self.opt_step(p)
    self.i += 1

  def opt_step(self,p):
    p -= p.grad * self.lr 
  
  def reg_step(self, p):     # regularization
    if self.wd != 0:
      p *= self.lr*self.wd
  def zero_grad(self):
    for p in self.params:
      p.grad.data.zero_()

class Momentum(SGD):
  def __init__(self, params , lr , wd=0. , mom = .9):
    super().__init__(params , lr=lr,wd=wd)
    self.mom = mom
  
  def opt_step(self , p):
    if not hasattr(p , 'grad_avg'):
      p.grad_avg = torch.zeros_like(p.grad)
      p.grad_avg = p.grad_avg * self.mom + p.grad* (1- self.mom)
      p -= self.lr * p.grad_avg


class RMSProp(SGD):
    def __init__(self, params, lr, wd=0., sqr_mom=0.99, eps=1e-5):
        super().__init__(params, lr=lr, wd=wd)
        self.sqr_mom,self.eps = sqr_mom,eps

    def opt_step(self, p):
        if not hasattr(p, 'sqr_avg'): p.sqr_avg = p.grad**2
        p.sqr_avg = p.sqr_avg*self.sqr_mom + p.grad**2*(1-self.sqr_mom)
        p -= self.lr * p.grad/(p.sqr_avg.sqrt() + self.eps)

class Adam(SGD):
    def __init__(self, params, lr, wd=0., beta1=0.9, beta2=0.99, eps=1e-5):
        super().__init__(params, lr=lr, wd=wd)
        self.beta1,self.beta2,self.eps = beta1,beta2,eps

    def opt_step(self, p):
        if not hasattr(p, 'avg'): p.avg = torch.zeros_like(p.grad.data)
        if not hasattr(p, 'sqr_avg'): p.sqr_avg = torch.zeros_like(p.grad.data)
        p.avg = self.beta1*p.avg + (1-self.beta1)*p.grad
        unbias_avg = p.avg / (1 - (self.beta1**(self.i+1)))
        p.sqr_avg = self.beta2*p.sqr_avg + (1-self.beta2)*(p.grad**2)
        unbias_sqr_avg = p.sqr_avg / (1 - (self.beta2**(self.i+1)))
        p -= self.lr * unbias_avg / (unbias_sqr_avg + self.eps).sqrt()

