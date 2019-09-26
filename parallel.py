import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
'''
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-parallel-tutorial-py
put the model on GPU
device = torch.device('cuda:0')
model.to(device)

copy tensor to GPU
mytensor = my_tensor.to(device)

multiple GPU
model = nn.DataParallel(model)
'''

# Parameters and DataLoaders

input_size = 5
out_put = 2

