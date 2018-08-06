Resnet详解
==========

# 一. 概述

> 随着网络的加深，出现训练集准确率下降的现象，可以确定这不是由于Overfit过拟合造成的(过拟合的情况训练集应该准确率很高)；所以作者针对这个问题提出了一种全新的网络，叫深度残差网络即Resnet，它允许网络尽可能的加深，将层变为学习关于层输入的残差函数，而不是学习未参考的函数。ResNet在2015年被提出，在ImageNet比赛classification任务上获得第一名，因为它“简单与实用”并存，之后很多方法都建立在ResNet50或者ResNet101的基础上完成的，检测、分割、识别等领域都纷纷使用ResNet，Alpha zero也使用了ResNet。

# 二. 网络结构

## (一) 残差

> 论文中，作者提到，考虑将期望的底层映射表示为H(x)，网络采用多个堆叠的非线性层渐进地逼近这个复杂的函数，那么就等同于渐进地逼近残差函数，使其为0，残差函数F(x):=H(x)−x。因此，原始的映射重写为F(x)+x。虽然两种形式都能渐进地逼近期望函数，但是逼近残差映射比原始的、未参考的映射更容易优化学习。

## (二) 实现

> 公式F(x)+x可以通过带有“快捷连接——shortcut connection”前向神经网络实现，如图所示。

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/1.png)

> 图中包含两部分：identity mapping——指的是图中弯曲的曲线，residual mapping——指的是直线传播的那部分，最终的输出y=F(x)+x。

> 关于shortcut connection，文中提出了两种方式：第一种是 y=F(X,Wi)+x，其中，对于两层卷积 F(X,Wi)=W2·σ(W1· X)，对于多层卷积F(X,Wi)=W3·σ(W2·σ(W1· X))，做相应调整即可，如图所示：

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/2.png)

> 该网络相对于普通网络没有增加任何参数，这两种结构分别针对Resnet34（左图）和Resnet50/101/152（右图），一般称整个结构为一个"building block"，其中右图又称为"bottleneck desigh"，其目的是为了减少参数的数目，第一个1x1的卷积把256维channel降到64维，然后在最后通过1x1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632，而不使用bottleneck的话就是两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648，差了16.94倍。 对于常规ResNet，可以用于34层或者更少的网络中，对于Bottleneck Design的ResNet通常用于更深的如101这样的网络中，目的是减少计算和参数量（实用目的）。F(X,Wi)

> 第二种是 y=F(X,Wi)+Ws· X，当输入输出的channels个数不一致时，可以采用Ws· X做调整，如图所示：

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/3.png)

> 此种情况主要出现在图中不同颜色色块连接的虚线处，因为此时涉及影像size的变化，因此需要做调整。
![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/4.jpg)

## (三) 设计

> 网络的设计与普通网络的不同在于：每一个building block中增加了Batch Normalization即BN————BN主要用于卷积层之后，激活函数之前。没有全连接层和dropout层。最终卷积结果采用全局池化，然后展平，softmax输出结果。

## (四) 详解

> 论文中Resnet网络结构如图所示，此处对Resnet34和Resnet50做详细解释，对每一层结构进行剖析。
![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/5.png)

### 1. Resnet 34

> 输入影像 224 x 224 x 3

> 第一层卷积 conv1：卷积核7 x 7 x 3 x 64 步长stride=2  卷积方式same（即zero padding）， 影像变为 112 x 112 x 64

> max pool：3 x 3 步长stride=2  卷积方式same，影像变为 56 x 56 x 64

> 第二层卷积 conv2：卷积核3 x 3 x 3 x 64  卷积方式same  共2 x 3 = 6 层 3个building block，每一个building block如图所示，

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/6.png)

> 3个building block串联起来如图所示，影像变为 56 x 56 x 64

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/7.png)
![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/8.png)

> 第三层卷积 conv3：卷积核3 x 3 x 3 x 128  卷积方式same  共2 x 4 = 8 层 4个building block，如图所示，影像变为 28 x 28 x 128

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/9.png)
![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/10.png)

> 第四层卷积 conv4：卷积核3 x 3 x 3 x 256  卷积方式same  共2 x 6 = 12 层 6个building block，形式同上，影像变为 14 x 14 x 256

> 第五层卷积 conv5：卷积核3 x 3 x 3 x 512  卷积方式same  共2 x 3 = 6 层 3个building block，形式同上，影像变为 7 x 7 x 512

> 全局average pool：影像变为 1 x 1 x 512

> 展平：影像变为 1 x 512

> 全连接层 1000，softmax，之后分类：影像变为 1 x 1000，此处为分类数，由于数据集是Imagenet，有1000类，可根据自己的数据集进行调整

### 2. Resnet 50

> 输入影像 224 x 224 x 3

> 第一层卷积 conv1：卷积核7 x 7 x 3 x 64 步长stride=2  卷积方式same（即zero padding）， 影像变为 112 x 112 x 64

> max pool：3 x 3 步长stride=2  卷积方式same，影像变为 56 x 56 x 64

> 第二层卷积 conv2：卷积核3 x 3 x 3 x 64  卷积方式same  共3 x 3 = 9 层 3个building block，如图所示，影像变为 56 x 56 x 256

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/11.png)

> 第三层卷积 conv3：卷积核3 x 3 x 3 x 128  卷积方式same  共3 x 4 = 12 层 4个building block，如图所示，影像变为 28 x 28 x 512

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/12.png)

> 第四层卷积 conv4：卷积核3 x 3 x 3 x 256  卷积方式same  共3 x 6 = 18 层 6个building block，形式同上，影像变为 14 x 14 x 1024

> 第五层卷积 conv5：卷积核3 x 3 x 3 x 512  卷积方式same  共3 x 3 = 9 层 3个building block，形式同上，影像变为 7 x 7 x 2048

> 全局average pool：影像变为 1 x 1 x 2048

> 展平：影像变为 1 x 2048

> 全连接层 1000，softmax，之后分类：影像变为 1 x 1000，此处为分类数，由于数据集是Imagenet，有1000类，可根据自己的数据集进行调整

# 三. 代码

> 利用MNIST数据集，构建Resnet34网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下

```python

```
