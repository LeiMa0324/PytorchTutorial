import torch

import numpy as np

#https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py#

#======================tensor 初始化========================#
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)      # tensor 从numpy获取数据
print(np_data)
print(torch_data)


x = torch.empty(5, 3)       # 空tensor，注意与zeros的区别
print(x)


x = torch.rand(5, 3)    # 随机生成tensor
print(x)


x = torch.zeros(5, 3,dtype= torch.long)     # 创建零tensor
print(x)


x = torch.tensor([5., 3])   # tensor，使用一个存在的list生成tensor
print(x)


x = x.new_ones(5, 3, dtype= torch.double)   # 使用一个已有的tensor，生成一个一样大小的one tensor
print(x)


x = torch.randn_like(x, dtype= torch.float) # rewrite dtype 保留其他属性的random tensor
print(x)
print(x.size())     # 打印tensor的size，为tuple类型


#======================Operation tensor运算========================#

# 加法一
y = torch.rand(5, 3)
print(x+y)

# 加法二
print(torch.add(x,y))

# 加法三
result = torch.empty(5, 3)
torch.add(x, y , out=result)    # 将结果赋给result
print(result)

# 加法四
y.add_(x)       #直接将x加到y上，所有in-place的函数都以_结尾，将会改变原有tensor
print(y)

# 切片操作，所有行，第一列
print(y[:,1])

# resize,reshape
x = torch.rand(4, 4)
y = x.view(16)      # view = np.reshape 将一个4*4的tensor转换为16*1
z = x.view(-1,8)    # -1代表该列根据第二个参数决定，16、8 =2, z为2*8的tensor
print(x.size())
print(y.size())
print(z.size())

# 将1维的tensor转化为python数值
x = torch.randn(1)
print(x)
print(x.item())

#======================tensor与numpy转换========================#
a = torch.ones(5)
print(a)

b = a.numpy()   # x.numpy() 将tensor转换为numpy，两个为指针的关系，a改变同时b也改变
print(b)

a.add_(1)
print(b)


a = np.ones(5)  # 将numpy转化为tensor，二者也是指针的关系，改变一个的同时改变另一个
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#======================Device========================#
if torch.cuda.is_available():
    device = torch.device("cuda")   # a cuda device object
    y = torch.ones_like(x, device= device)      #在device上创造一个和x一样size的ones tensor(ones_like函数),
    x = x.to(device)                    # 或者将已有tensor发送至device上
    z = x + y
    print(z)
    print(z.to("cpu",torch.double))     # 将z发送至cpu上，并指定torch类型为double