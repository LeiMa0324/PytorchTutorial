import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable


# Step1: 先给每个词一个index
word_to_ix = {"hello":0, "world":1}

# Step2: 建立一个embeding，词典大小为2，每个词向量的纬度为5
embeds = nn.Embedding(2, 5)

# Step3: 将hello的index转化为tensor
helloidx = torch.LongTensor([word_to_ix["hello"]])
print(helloidx)

# #
# helloidx = torch.tensor(helloidx)
# print(helloidx)

# Step3：对hello的index进行embedding
hello_embed = embeds(helloidx)
print(hello_embed)
