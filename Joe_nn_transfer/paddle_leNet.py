import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Conv2DTranspose
from paddle.fluid.dygraph.base import to_variable
# K.set_image_data_format('channels_first')
# 定义 LeNet 网络结构
class LeNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes=1):
        super(LeNet, self).__init__(name_scope)

        self.conv1 = Conv2D(num_channels=1, num_filters=6, filter_size=5, act='relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        
        # self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='relu')
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分裂标签的类别数
        self.fc1 = Linear(input_dim=16*5*5, output_dim=120, act='relu')
        self.fc2 = Linear(input_dim=120, output_dim=84, act='relu')
        self.fc3 = Linear(input_dim=84, output_dim=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    