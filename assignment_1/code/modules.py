"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.size_in = in_features
        self.size_out = out_features
        self.input = None

        self.params = {'weight': np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features)),
                       'bias': np.zeros(out_features)}
        self.grads = {'weight': np.zeros((out_features, in_features)),
                      'bias': np.zeros(out_features)}
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = np.zeros((self.size_out, self.size_in))
        self.grads['bias'] = np.zeros(self.size_out)

        self.input = x
        out = x @ self.params['weight'].T + self.params['bias']
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout @ self.params['weight']
        self.grads['weight'] = dout.T @ self.input
        self.grads['bias'] = np.sum(dout, axis=0)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = x.copy()
        out = x.copy()
        out[out <= 0] = 0
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout
        dx[self.input < 0] = 0
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        b = x.max(axis=1).reshape(-1, 1)
        y = np.exp(x - b)
        out = y / y.sum(axis=1).reshape(-1, 1)
        self.out = out.copy()
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size = dout.shape[0]
        dout = dout.reshape(batch_size, 1, -1)
        x_n = self.out.reshape(batch_size, 1, -1)
        dx_tilda = np.apply_along_axis(np.diag, 1, self.out) - x_n.transpose((0, 2, 1)) @ x_n
        dx = (dout @ dx_tilda).reshape(batch_size, -1)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.sum(-np.log(x[y == 1])) / x.shape[0]
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = - np.divide(y, x) / x.shape[0]
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
