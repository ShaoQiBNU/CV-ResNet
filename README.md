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

> 第二层卷积 conv2：卷积核3 x 3 x 64 x 64  卷积方式same  共2 x 3 = 6 层 3个building block，每一个building block如图所示，

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/6.png)

> 3个building block串联起来如图所示，影像变为 56 x 56 x 64

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/7.png)
![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/8.png)

> 第三层卷积 conv3：卷积核3 x 3 x 64 x 128  卷积方式same  共2 x 4 = 8 层 4个building block，如图所示，影像变为 28 x 28 x 128

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/9.png)
![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/10.png)

> 第四层卷积 conv4：卷积核3 x 3 x 128 x 256  卷积方式same  共2 x 6 = 12 层 6个building block，形式同上，影像变为 14 x 14 x 256

> 第五层卷积 conv5：卷积核3 x 3 x 256 x 512  卷积方式same  共2 x 3 = 6 层 3个building block，形式同上，影像变为 7 x 7 x 512

> 全局average pool：影像变为 1 x 1 x 512

> 展平：影像变为 1 x 512

> 全连接层 1000，softmax，之后分类：影像变为 1 x 1000，此处为分类数，由于数据集是Imagenet，有1000类，可根据自己的数据集进行调整

### 2. Resnet 50

> 输入影像 224 x 224 x 3

> 第一层卷积 conv1：卷积核7 x 7 x 3 x 64 步长stride=2  卷积方式same（即zero padding）， 影像变为 112 x 112 x 64

> max pool：3 x 3 步长stride=2  卷积方式same，影像变为 56 x 56 x 64

> 第二层卷积 conv2：卷积核1 x 1   3 x 3   1 x 1  卷积方式same  共3 x 3 = 9 层 3个building block，如图所示，影像变为 56 x 56 x 256

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/11.png)

> 第三层卷积 conv3：卷积核1 x 1   3 x 3   1 x 1  卷积方式same  共3 x 4 = 12 层 4个building block，如图所示，影像变为 28 x 28 x 512

![image](https://github.com/ShaoQiBNU/Resnet/blob/master/images/12.png)

> 第四层卷积 conv4：卷积核1 x 1   3 x 3   1 x 1  卷积方式same  共3 x 6 = 18 层 6个building block，形式同上，影像变为 14 x 14 x 1024

> 第五层卷积 conv5：卷积核1 x 1   3 x 3   1 x 1  卷积方式same  共3 x 3 = 9 层 3个building block，形式同上，影像变为 7 x 7 x 2048

> 全局average pool：影像变为 1 x 1 x 2048

> 展平：影像变为 1 x 2048

> 全连接层 1000，softmax，之后分类：影像变为 1 x 1000，此处为分类数，由于数据集是Imagenet，有1000类，可根据自己的数据集进行调整

# 三. 代码

> 利用MNIST数据集，构建Resnet34网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

```python
########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_sets",one_hot=True)

########## set net hyperparameters ##########
learning_rate=0.0001

epochs=20
batch_size_train=128
batch_size_test=100

display_step=20

########## set net parameters ##########
#### img shape:28*28 ####
n_input=784 

#### 0-9 digits ####
n_classes=10

########## placeholder ##########
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])


##################### build net model ##########################

######### identity_block #########
def identity_block(inputs,filters,kernel,strides):
    '''
    identity_block: 两层的恒等残差块，影像输入输出的height和width保持不变，channel发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 两层恒等残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1,f2=filters
    k1,k2=kernel
    s1,s2=strides


    ######## shortcut 第一种规则，影像输入输出的height和width保持不变，输入直接加到卷积结果上 ########
    inputs_shortcut=inputs


    ######## first identity block 第一层恒等残差块 ########
    #### conv ####
    layer1=tf.layers.conv2d(inputs,filters=f1,kernel_size=k1,strides=s1,padding='SAME')

    #### BN ####
    layer1=tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1=tf.nn.relu(layer1)


    ######## second identity block 第二层恒等残差块 ########
    #### conv ####
    layer2=tf.layers.conv2d(layer1,filters=f2,kernel_size=k2,strides=s2,padding='SAME')

    #### BN ####
    layer2=tf.layers.batch_normalization(layer2)
    

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out=tf.add(inputs_shortcut,layer2)

    ######## relu ########
    out=tf.nn.relu(out)
    
    return out


######## convolutional_block #########
def convolutional_block(inputs,filters,kernel,strides):
    '''
    convolutional_block: 两层的卷积残差块，影像输入输出的height、width和channel均发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 两层的卷积残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1,f2=filters
    k1,k2=kernel
    s1,s2=strides


    ######## shortcut 第二种规则，影像输入输出height和width发生变化，需要对输入做调整 ########
    #### conv ####
    inputs_shortcut=tf.layers.conv2d(inputs,filters=f1,kernel_size=1,strides=s1,padding='SAME')

    #### BN ####
    inputs_shortcut=tf.layers.batch_normalization(inputs_shortcut)


    ######## first convolutional block 第一层卷积残差块 ########
    #### conv ####
    layer1=tf.layers.conv2d(inputs,filters=f1,kernel_size=k1,strides=s1,padding='SAME')

    #### BN ####
    layer1=tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1=tf.nn.relu(layer1)


    ######## second convolutional block 第二层卷积残差块 ########
    #### conv ####
    layer2=tf.layers.conv2d(layer1,filters=f2,kernel_size=k2,strides=s2,padding='SAME')

    #### BN ####
    layer2=tf.layers.batch_normalization(layer2)


    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out=tf.add(inputs_shortcut,layer2)


    ######## relu ########
    out=tf.nn.relu(out)
    
    return out


######### Resnet 34 layer ##########
def Resnet34(x,n_classes):

    ####### reshape input picture ########
    x=tf.reshape(x,shape=[-1,28,28,1])


    ####### first conv ########
    #### conv ####
    conv1=tf.layers.conv2d(x,filters=64,kernel_size=7,strides=2,padding='SAME')

    #### BN ####
    conv1=tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1=tf.nn.relu(conv1)


    ####### max pool ########
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### second conv ########
    #### convolutional_block 1 ####
    conv2=convolutional_block(pool1,filters=[64,64],kernel=[3,3],strides=[1,1])

    #### identity_block 2 ####
    conv2=identity_block(conv2,filters=[64,64],kernel=[3,3],strides=[1,1])
    conv2=identity_block(conv2,filters=[64,64],kernel=[3,3],strides=[1,1])


    ####### third conv ########
    #### convolutional_block 1 ####
    conv3=convolutional_block(conv2,filters=[128,128],kernel=[3,3],strides=[2,1])

    #### identity_block 3 ####
    conv3=identity_block(conv3,filters=[128,128],kernel=[3,3],strides=[1,1])
    conv3=identity_block(conv3,filters=[128,128],kernel=[3,3],strides=[1,1])
    conv3=identity_block(conv3,filters=[128,128],kernel=[3,3],strides=[1,1])


    ####### fourth conv ########
    #### convolutional_block 1 ####
    conv4=convolutional_block(conv3,filters=[256,256],kernel=[3,3],strides=[2,1])
    
    #### identity_block 5 ####
    conv4=identity_block(conv4,filters=[256,256],kernel=[3,3],strides=[1,1])
    conv4=identity_block(conv4,filters=[256,256],kernel=[3,3],strides=[1,1])
    conv4=identity_block(conv4,filters=[256,256],kernel=[3,3],strides=[1,1])
    conv4=identity_block(conv4,filters=[256,256],kernel=[3,3],strides=[1,1])
    conv4=identity_block(conv4,filters=[256,256],kernel=[3,3],strides=[1,1])


    ####### fifth conv ########
    #### convolutional_block 1 ####
    conv5=convolutional_block(conv4,filters=[512,512],kernel=[3,3],strides=[2,1])
    
    #### identity_block 2 ####
    conv5=identity_block(conv5,filters=[512,512],kernel=[3,3],strides=[1,1])
    conv5=identity_block(conv5,filters=[512,512],kernel=[3,3],strides=[1,1])


    ####### 全局平均池化 ########
    #pool2=tf.nn.avg_pool(conv5,ksize=[1,7,7,1],strides=[1,7,7,1],padding='VALID')


    ####### flatten 影像展平 ########
    flatten = tf.reshape(conv5, (-1, 1*1*512))


    ####### out 输出，10类 可根据数据集进行调整 ########
    out=tf.layers.dense(flatten,n_classes)


    ####### softmax ########
    out=tf.nn.softmax(out)

    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred=Resnet34(x,n_classes)

#### loss 损失计算 ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size_test):
        batch_x,batch_y=mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))

```

> 利用MNIST数据集，构建Resnet50网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

```python

```
