# PyTorch_implementation.
This repository contains an implementation of important modules in PyTorch from scratch, covering various aspects of deep learning. It includes modules such as **nn.Module nn.layers, PyTorch hooks, batch and layer normalization, optimizer, learning rate scheduler, activation and weight initialization, dataset, dataloader**, and a **flexible learner** function inspired by **Hugging Face**.


- [Tensor_&_broadcating.ipynb](Tensor_&_broadcating.ipynb)   - Radnom number & Broadcating 
- [forward_&_backward.ipynb](forward_&_backward.ipynb)   - Backpropgation , nn.Module and AutoGrad
- [NN_Training.ipynb](NN_Training.ipynb)   - cross entropy loss , Parameters , Optim , Dataset , Dataloader
- [dataset.ipynb](dataset.ipynb)   - Dataset , Dataloader , Grid image repersentation
- [dataset.ipynb](dataset.ipynb)   - Dataset , Dataloader , Grid image repersentation
- [activation.py](activation.py). - GeneralReLU , Pytorch Hook , network weight initlization  - xavier , kaiming_normal ,lsuv_init ,Layernorm , BatchNorm.
- [Library.py](Library.py)   - Helper funcation (memory cleaning , to_cpu ,to_device , show_image)
- [learner.py](learner.py) - framework to train and evaluate model and flexiable to callbacks
- [callbacks,py](callbacks.py) - CompletionCB , SingleBatchCB , MetricsCB , DeviceCB , ProgressCB , TrainCB ,HooksCallback , BatchSchedCB ,WandbCB.
- [optimizer.py](optimizer.py) - SGD , Momentum , RMSProp , Adam ,
