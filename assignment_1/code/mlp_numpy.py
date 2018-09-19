"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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
        self.layers = []
        self.sz_input = n_inputs

        previous_size = n_inputs
        for current_size in n_hidden:
            self.layers.append(LinearModule(previous_size, current_size))
            self.layers.append(ReLUModule())
            previous_size = current_size

        self.layers.append(LinearModule(previous_size, n_classes))

        self.output_activation = SoftMaxModule()
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
        if x.ndim == 4:  # ensure right shape of input
            x = x.reshape(-1, self.sz_input)
        elif x.ndim == 3:
            x = x.reshape(-1)

        for layer in self.layers:
            x = layer.forward(x)

        out = self.output_activation.forward(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dout = self.output_activation.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        ########################
        # END OF YOUR CODE    #
        #######################

        return
