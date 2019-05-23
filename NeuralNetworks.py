import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py#

'''
nn package
nn 依赖于autograd定义网络并执行微分。
nn.Module包括网络层
nn.forward()方法输出

在一个网络内，仅需要定义forward function,当使用autograd时，backward function自动定义
'''

#======================Define the network========================#

class Net(nn.Module):

    def __init__(self):         # 仅定义layers
        super(Net, self).__init__()     #调用父类的init函数
        self.conv1 = nn.Conv2d(1, 6, 5)     # input channel, output channel, 5*5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)   #input size，outputsize
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):       # 定义网络结构和output怎么得到,forward 方法
        x = F.max_pool2d(F.relu(self.conv1(x)),(2, 2))      ## Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x        # 神经网络最后的输出

    def num_flat_features(self, x):
        size = x.size()[1:]     # size()返回一个tuple
        num_features = 1
        for s in size:  # 得到该size内所有元素的个数
            num_features *=s
        return num_features

net = Net()
print(net)

params = list(net.parameters())     #返回net中可以更新的参数列表，其中每个元素是每一层的param的列表
print(len(params))
print(params[0].size())     # 第一层，即conv1的weights

#======================Processing input and backward========================#
input = torch.randn(1, 1, 32, 32)   # 32*32的input
out = net(input)        # 经过network后的输出
print(out)

net.zero_grad()     # zero the gradients buffer
out.backward(torch.randn(1, 10))    # backprop with random gradients, 10 维的gradients


#======================Loss Function========================#
'''
nn.MSELoss: 计算mean square error
'''

output = net(input)
target = torch.randn(10)    # 定义损失函数中的target
target = target.view(1, -1)     # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)        #定义损失函数
print(loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

#======================back prop========================#
'''
1. 清空所有的gradients
2. loss.backward()
'''

net.zero_grad()     # 清空gradients
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)      # 此处全为0

loss.backward()         # back propagation

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)  # conv1的gradient

#======================update the weights========================#
'''
simplest update: SGD, stochastic gradient descent
'''
learning_rate = 0.01
for f in net.parameters():      # 对于所有的parameter
    f.data.sub_(f.grad.data * learning_rate)    #自减一个learningrate*grad

'''
其他optimizer使用torch.optim
'''
optimizer = optim.SGD(net.parameters(),lr = 0.01)   #创建优化器

#the training loop
for i in range(10):
    optimizer.zero_grad()       # 清空gradient buffer
    output = net(input)         # 获得输出
    loss = criterion(output, target)    # 定义loss
    loss.backward()     # backprop
    optimizer.step()    # 对parameters做更新
    print("net.conv1.bias.grad in %d iteration"%i)
    print(net.conv1.bias.grad)