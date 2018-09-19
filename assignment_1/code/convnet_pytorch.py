"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()
        self.input_size = n_channels
        self.output_size = n_classes
        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(self.input_size, 64, kernel_size=3, stride=1, padding=1))  # conv1
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # maxpool1

        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))  # conv2
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # maxpool2

        self.layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))  # conv3_a
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))  # conv3_b
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # maxpool3

        self.layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))  # conv4_a
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))  # conv4_b
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # maxpool4

        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))  # conv5_a
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))  # conv5_b
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # maxpool5

        self.layers.append(nn.AvgPool2d(kernel_size=1, stride=1, padding=0))  # avgpool
        self.layers.append(nn.Linear(in_features=512, out_features=self.output_size))  # linear
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = x.reshape(x.shape[0], -1)  # make sure input to Linear layer has correct shape
            x = layer.forward(x)

        out = x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
