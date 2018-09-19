"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.sz_input = n_inputs

        previous_size = n_inputs
        if len(n_hidden) != 0:
            for idx, current_size in enumerate(n_hidden):
                # Architecture used for default MLP implementation
                # self.layers.append(nn.Linear(previous_size, current_size))
                # self.layers.append(nn.ReLU())
                # previous_size = current_size

                # Improved architecture for increased performance; uses weight norm and dropout
                self.layers.append(nn.utils.weight_norm(nn.Linear(previous_size, current_size)))
                self.layers.append(nn.ReLU())
                if idx < 5:
                    self.layers.append(nn.Dropout(0.2))
                previous_size = current_size

        self.layers.append(nn.Linear(previous_size, n_classes))
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
        x = x.reshape(-1, self.sz_input)

        for idx, layer in enumerate(self.layers):
            x = layer(x)

        out = x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
