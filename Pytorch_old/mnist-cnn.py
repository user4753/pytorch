import torch
import torch.nn as nn
inputs =torch.Tensor(1,1,28,28)
inputs
inputs.shape
conv1 =nn.Conv2d(1,32,3,padding=1)
pool = nn.MaxPool2d(2)
pool
conv2 = nn.Conv2d(32,64,3,padding=1)
conv1
conv2
pool
out =conv1(inputs)
out.shape
out = pool(out)
out.shape
out= conv2(out)
out.shape
out = pool(out)
out.shape
out.size(0)
out.size(1)
out.size(2)
out.size(3)
out =out.view(out.size(0),-1)
out.shape
fc = nn.Linear(3136,10)
out = fc(out)
out
out.shape