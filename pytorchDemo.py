import torch  # 导入pytorch
from torch import nn, optim  # 导入神经网络与优化器对应的类
import torch.nn.functional as F
from torchvision import datasets, transforms ## 导入数据集与数据预处理的方法

# 数据预处理：标准化图像数据，使得灰度数据在-1到+1之间
# transforms.ToTensor() 把图像变为张量
# transforms.Normalize((0.5,), (0.5,))让他标准化，标准化：就是减去平均值再除以标准差，让他处于-1到1之间

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# 下载Fashion-MNIST训练集数据，并构建训练集数据载入器trainloader,每次从训练集中载入64张图片，每次载入都打乱顺序
#使用torchvision的datasets方法 下载FashionMNIST
#trainset = datasets.FashionMNIST('dataset/', download=True, train=True, transform=transform)
#'dataset/' 是存储路径
#download=True 开启下载  train=True =下载的是训练集
#transform=transform     transform是我们上面归一化的变化
trainset = datasets.FashionMNIST('dataset/', download=True, train=True, transform=transform)
#DataLoader装载
#trainset 我们下载的图片
#batch_size=64 意味着64张照片，
#shuffle=True 打乱顺序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载Fashion-MNIST测试集数据，并构建测试集数据载入器trainloader,每次从测试集中载入64张图片，每次载入都打乱顺序
testset = datasets.FashionMNIST('dataset/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


#迭代器
image, label = next(iter(trainloader))

# image图片中有64张图片，我们查看索引为2的图片
imagedemo = image[3]
imagedemolabel = label[3]

imagedemo = imagedemo.reshape((28,28))

import matplotlib.pyplot as plt

plt.imshow(imagedemo)

labellist = ['T恤','裤子','套衫','裙子','外套','凉鞋','汗衫','运动鞋','包包','靴子']
print(f'这张图片对应的标签是 {labellist[imagedemolabel]}')



# 2 搭建并训练四层全连接神经网络

from torch import nn, optim
import torch.nn.functional as F

#初始化神经网络
class Classifier(nn.Module):

    # 初始化神经网络
    def __init__(self):
        super().__init__()
        #神经网络的输入 28 * 28=784 个像素
        #隐层1
        self.fc1 = nn.Linear(784, 256)
        #隐层2
        self.fc2 = nn.Linear(256, 128)
        #隐层3
        self.fc3 = nn.Linear(128, 64)
        #输出层
        self.fc4 = nn.Linear(64, 10)


    #对每一层的神经网络进行激活函数处理
    def forward(self, x):
        # make sure input tensor is flattened
        #定义一个正向传播的函数
        x = x.view(x.shape[0], -1)

        # 使用relu函数进行数据处理
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #log_softmax 归一化
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

        # 对上面定义的Classifier类进行实例化
        model = Classifier()

        # 定义损失函数为负对数损失函数
        #通过求导求出参数后，使用梯度下降法 ，让损失函数降低
        criterion = nn.NLLLoss()

        # 优化方法为Adam梯度下降方法，学习率为0.003
        #Adam梯度下降方法
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        # 对训练集的全部数据学习15遍，这个数字越大，训练时间越长
        epochs = 15

        # 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
        train_losses, test_losses = [], []

        print('开始训练')
        for e in range(epochs):
            running_loss = 0

            # 对训练集中的所有图片都过一遍
            for images, labels in trainloader:
                # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
                #这是pytorch的潜规则
                optimizer.zero_grad()

                # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
                # 正向传播
                log_ps = model(images)
                #算出预测误差和真实误差的区别
                loss = criterion(log_ps, labels)
                #对每个误差进行反向传播
                loss.backward()
                #在进行优化迭代
                optimizer.step()
                #再把每个误差加起来
                running_loss += loss.item()

            # 每次学完一遍数据集，都进行以下测试操作
            else:
                test_loss = 0
                accuracy = 0
                # 测试的时候不需要开自动求导和反向传播
                with torch.no_grad():
                    # 关闭Dropout
                    model.eval()

                    # 对测试集中的所有图片都过一遍
                    for images, labels in testloader:
                        # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
                        log_ps = model(images)
                        test_loss += criterion(log_ps, labels)
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)

                        # 等号右边为每一批64张测试图片中预测正确的占比
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                # 恢复Dropout
                model.train()
                # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))

                print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
                      "训练误差: {:.3f}.. ".format(running_loss / len(trainloader)),
                      "测试误差: {:.3f}.. ".format(test_loss / len(testloader)),
                      "模型分类准确率: {:.3f}".format(accuracy / len(testloader)))