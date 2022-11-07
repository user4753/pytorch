import torch
import torch.nn as nn
conv = nn.Conv2d(1,1,11,stride=4,padding=0)
conv
inputs = torch.Tensor(1,1,227,227)
inputs.shape
out = conv(inputs)
out.shape
conv =nn.Conv2d(1,1,7,stride=2,padding=0)
inputs = torch.Tensor(1,1,64,64)
out =conv(inputs)
out.shape
conv =nn.Conv2d(1,1,5,stride=1,padding=2)
inputs = torch.Tensor(1,1,32,32)
out =conv(inputs)
out.shape
inputs = torch.Tensor(1,1,32,64)
conv =nn.Conv2d(1,1,5,stride=1,padding=0)
out =conv(inputs)
out.shape
inputs = torch.Tensor(1,1,64,32)
conv =nn.Conv2d(1,1,3,stride=1,padding=1)
out =conv(inputs)
out.shape


inputs = torch.Tensor(1,1,28,28)
conv1 =nn.Conv2d(1,5,5)
pool = nn.MaxPool2d(2)
out =conv1(inputs)
out2 = pool(out)
out.size()
out2.size()