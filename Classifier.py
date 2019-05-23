import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from torch.nn import functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
对数据的操作：
1. 将数据存储为numpy数组
2. 将numpy数组转化为tensor

CIFAR10 datasets: 3 通道的彩色图片，32*32，10 个类别的物体 3*32*32

steps:
1. load并normalize CIFAR10 dataset using torchvision
2. 定义CNN
3. 定义loss function
4. train on training data
5. test on test data
'''

#=====================Load and Normalize dataset========================#

# compose函数，将几个transform结合在一起
# 转换一：转为tensor，转换二：归一化, 第一个参数为需要normalize的dataset的mean，第二个为std，3个channel所以是三维，从[0-1]转换为[-1,1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train = True，则下载trainset，否则下载testset
trainset = torchvision.datasets.CIFAR10(root='./data', train= True, download= True, transform= transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root= './data', train= False, download= True, transform= transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers =2)

classes =['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

#=====================fuinctions to show images=======================#

# 此处img是一个tensor
def imshow(img):
    img = img/2 + 0.5   # unnormalize，从[-1,1]转换回[0,1]
    npimg = img.numpy()     #将tensor转换为numpy数组
    '''
    pytorch内，order of dimension是 channel*width*height
    matplot内，order是 width*height*channel
    所以需要转置
    '''
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get random image
dataiter = iter(trainloader)    # get an iterator from the data loader
images, labels = dataiter.next()    # 获取一组图片和label

# show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#============================Define a CNN============================#
'''
conv1d是一个数组
conv2d是一个矩阵（常用卷积）
conv3d是一个tensor
'''
class Net(nn. Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        输出有6个channel，即该层有6个神经元，每个卷积核的大小为5*5
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)     #input channel, output channel,kernel size
        self.pool = nn.MaxPool2d(2, 2)
        '''
        第二层卷积层有16个神经元，每个卷积核大小为5*5
        '''
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)   # 输入的size，输出size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # conv->relu->pooling
        x = self.pool(F.relu(self.conv2(x)))    # conv->relu->pooling
        x = x.view(-1, 16*5*5)  # 将经过卷积的x压成16*5*5列的矩阵
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


#============================Define Loss Function============================#