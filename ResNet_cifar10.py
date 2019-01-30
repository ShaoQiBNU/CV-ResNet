######################## load packages ###########################
import tensorflow as tf
import numpy as np
import cifar10

######################## download dataset ###########################
cifar10.maybe_download_and_extract()

######################## load train and test data ###########################
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

######################## network paramaters ###########################
epochs = 20
train_batch_size = 128
test_batch_size = 100

learning_rate = 0.001
display_step = 20

########## set net parameters ##########
image_size = 32
num_channels = 3

#### 10 kind ####
n_classes = 10

########## placeholder ##########
x = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels])
y = tf.placeholder(tf.float32, [None, n_classes])


######################## random generate data ###########################
def random_batch(images, labels):
    '''
    :param images: 输入影像集
    :param labels: 输入影像集的label
    :return: batch data
    '''

    num_images = len(images)

    ######## 随机设定待选图片的id ########
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)

    ######## 筛选data ########
    x_batch = images[idx, :, :]
    y_batch = labels[idx, :]

    return x_batch, y_batch


##################### build net model ##########################
######### identity_block #########
def identity_block(inputs, filters, kernel, strides):
    '''
    identity_block: 两层的恒等残差块，影像输入输出的height和width保持不变，channel发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 两层恒等残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2 = filters
    k1, k2 = kernel
    s1, s2 = strides

    ######## shortcut 第一种规则，影像输入输出的height和width保持不变，输入直接加到卷积结果上 ########
    inputs_shortcut = inputs

    ######## first identity block 第一层恒等残差块 ########
    #### conv ####
    layer1 = tf.layers.conv2d(inputs, filters=f1, kernel_size=k1, strides=s1, padding='SAME')

    #### BN ####
    layer1 = tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    ######## second identity block 第二层恒等残差块 ########
    #### conv ####
    layer2 = tf.layers.conv2d(layer1, filters=f2, kernel_size=k2, strides=s2, padding='SAME')

    #### BN ####
    layer2 = tf.layers.batch_normalization(layer2)

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out = tf.add(inputs_shortcut, layer2)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


######## convolutional_block #########
def convolutional_block(inputs, filters, kernel, strides):
    '''
    convolutional_block: 两层的卷积残差块，影像输入输出的height、width和channel均发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 两层的卷积残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2 = filters
    k1, k2 = kernel
    s1, s2 = strides

    ######## shortcut 第二种规则，影像输入输出height和width发生变化，需要对输入做调整 ########
    #### conv ####
    inputs_shortcut = tf.layers.conv2d(inputs, filters=f1, kernel_size=1, strides=s1, padding='SAME')

    #### BN ####
    inputs_shortcut = tf.layers.batch_normalization(inputs_shortcut)

    ######## first convolutional block 第一层卷积残差块 ########
    #### conv ####
    layer1 = tf.layers.conv2d(inputs, filters=f1, kernel_size=k1, strides=s1, padding='SAME')

    #### BN ####
    layer1 = tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    ######## second convolutional block 第二层卷积残差块 ########
    #### conv ####
    layer2 = tf.layers.conv2d(layer1, filters=f2, kernel_size=k2, strides=s2, padding='SAME')

    #### BN ####
    layer2 = tf.layers.batch_normalization(layer2)

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out = tf.add(inputs_shortcut, layer2)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


######## ResNet CIFAR network #########
def Resnet_CIFAR(x, n_classes):

    ####### first conv ########
    #### conv ####
    conv1 = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='SAME')

    #### BN ####
    conv1 = tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1 = tf.nn.relu(conv1)


    ####### second conv ########
    #### convolutional_block 1 ####
    conv2 = convolutional_block(conv1, filters=[16, 16], kernel=[3, 3], strides=[1, 1])

    #### identity_block 2 ####
    conv2 = identity_block(conv2, filters=[16, 16], kernel=[3, 3], strides=[1, 1])


    ####### third conv ########
    #### convolutional_block 1 ####
    conv3 = convolutional_block(conv2, filters=[32, 32], kernel=[3, 3], strides=[2, 1])

    #### identity_block 3 ####
    conv3 = identity_block(conv3, filters=[32, 32], kernel=[3, 3], strides=[1, 1])


    ####### fourth conv ########
    #### convolutional_block 1 ####
    conv4 = convolutional_block(conv3, filters=[64, 64], kernel=[3, 3], strides=[2, 1])

    #### identity_block 5 ####
    conv4 = identity_block(conv4, filters=[64, 64], kernel=[3, 3], strides=[1, 1])


    ####### 全局平均池化 ########
    pool = tf.nn.avg_pool(conv4, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='VALID')

    ####### flatten 影像展平 ########
    flatten = tf.reshape(pool, (-1, 1 * 1 * 64))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)

    return out


########## define model, loss and optimizer ##########
#### model pred 影像判断结果 ####
pred = Resnet_CIFAR(x, n_classes)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(len(images_train) // train_batch_size):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = random_batch(images_train, labels_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(len(images_test) // test_batch_size):
        batch_x, batch_y = random_batch(images_test, labels_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
