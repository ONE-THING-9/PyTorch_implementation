from library import *



class CancelFitException(Exception):pass
class CancelBatchException(Exception):pass
class CancelEpochException(Exception):pass

class Callback():
  order =0

class CompletionCB(Callback):
  def before_fit(self,learn):
    self.count =0
  def after_batch(self , learn):
    self.count +=1
  def after_fit(self,learn):
    print(f"completed : {self.count} batches")

class SingleBatchCB(Callback):
  order =1
  def after_batch(self,learn):
    raise CancelFitException("SingleBatchCB")

  
class MetricsCB(Callback):
  def __init__(self, *ms ,**metrics):
    for o in ms:
      metrics[type(o).__name__] = o
    self.metrics = metrics
    self.all_metrics = copy(metrics)
    self.all_metrics['loss'] = self.loss = Mean()

  def _log(self,d):
    print(d)
  
  def before_fit(self,learn):
    learn.metrics= self
  def before_epoch(self,learn):
    [o.reset() for o in self.all_metrics.values()]
  
  def after_epoch(self,learn):
    log = {k:f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
    log['epoch'] = learn.epoch
    log['train'] = 'train' if learn.model.training else "eval"
    self._log(log)
  def after_batch(self,learn):
    x,y,*_ = to_cpu(learn.batch)
    for m in self.metrics.values():
      m.update(to_cpu(learn.preds) ,y)
      self.loss.update(to_cpu(learn.loss) , weight = len(x))

class DeviceCB(Callback):
    def __init__(self, device=def_device): fc.store_attr()
    def before_fit(self, learn):
        if hasattr(learn.model, 'to'): learn.model.to(self.device)
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)

class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False): self.plot = plot
    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn): learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses: self.mbar.update_graph([[fc.L.range(self.losses), self.losses],[fc.L.range(learn.epoch).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])
    
    def after_epoch(self, learn): 
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'): 
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph([[fc.L.range(self.losses), self.losses],[fc.L.range(learn.epoch+1).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])
     
class TrainCB(Callback):
    def __init__(self, n_inp=1): self.n_inp = n_inp
    def predict(self, learn): learn.preds = learn.model(*learn.batch[:self.n_inp])
    def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])
    def backward(self, learn): learn.loss.backward()
    def step(self, learn): learn.opt.step()
    def zero_grad(self, learn): learn.opt.zero_grad()

class Hook():
  def __init__(self , m , f):
    self.hook = m.register_forward_hook(partial(f , self))
  def remove(self):
    self.hook.remove()
  def __del__(self):
    self.remove()

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    acts = to_cpu(outp)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40,0,10))

class Hooks(list):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()
    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
    def remove(self):
        for h in self: h.remove()
     
class HooksCallback(Callback):
    def __init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None):
        fc.store_attr()
        super().__init__()
    
    def before_fit(self, learn):
        if self.mods: 
          mods=self.mods
        else: 
          mods = fc.filter_ex(learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training): self.hookfunc(*args, **kwargs)

    def after_fit(self, learn): self.hooks.remove()
    def __iter__(self): return iter(self.hooks)
    def __len__(self): return len(self.hooks)

def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)
def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()


class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop): super().__init__(append_stats, mod_filter)

    def color_dim(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)

    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for h in self:
            for i in 0,1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(fc.L.range(self))
    
class BatchTransformCB(Callback):
  def __init__(self,tfm,on_train=True , on_valid =True):
    fc.store_attr()
  
  def before_batch(self,learn):
    if (self.on_train and learn.training) or (self.on_val and not learn.training):
      learn.batch = self.tfm(learn.batch)

class BaseSchedCB(Callback):
  def __init__(self, sched):
    self.sched = sched
  def before_fit(self , learn):
    self.sched = self.sched(learn.opt)
  def _step(self , learn):
    if learn.training:
      self.sched.step()

class BatchSchedCB(BaseSchedCB):
  def after_batch(self , learn):
    self._step(learn)

class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn): self._step(learn)

class RecorderCB(Callback):
  def __init__(self ,**d):
    self.d =d
  def before_fit(self,learn):
    self.recs = {k:[] for k in self.d}
    self.pg = learn.opt.param_groups[0]

  def after_batch(self,learn):
    if not learn.training:
      return
    for k,v in self.d.items():
      self.recs[k].append(v(self))
  def plot(self):
    for k , v in self.recs.items():
      plt.plot(v , label =k)
      plt.legend()
      plt.show()

