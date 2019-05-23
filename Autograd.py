import torch
from torch import autograd


#https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py#

'''
对于一个tensor
requires_grad =  True, torch将track所有在改变量上的operation，默认为false
.backward() 将自动计算所有requires_grad打开的gradients，并保存在分别的a.grad内
.grad   保存所有的gradients
.detach()   停止track该变量

'''

x = torch.ones(2, 2, requires_grad= True)
print(x)
y = x + 2
print(y)    # grad_fn attribute 保存该tensor是怎么来的，如果是通过operation获得
print(y.grad_fn)
z = y*y*3
out = z.mean()  # 求z的均值
print(z,out)    # 查看z和y的grad_fn参数

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)      # 默认requires_grad为false
a.requires_grad = True
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)        # a打开requires_grad后，b的grad_fn生效

#======================Gradients========================#

'''
Jacobian matrix y向量关于x向量的偏导矩阵
'''
out.backward()      #因为out是一个标量，所以backward内无参数，等同于backward(torch.tensor(1.))
print(x.grad)       # 获得x关于标量out的Jacobian Matrix

x = torch.randn(3, requires_grad= True)
y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001],dtype=torch.float)
y.backward(v)

print(x.grad)   # 没看懂

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():       #在该with内，不track tensor的grads
    print((x**2).requires_grad)