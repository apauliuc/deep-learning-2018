import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

# noinspection PyUnresolvedReferences
class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormAutograd, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.neurons = n_neurons
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if input.shape[1] != self.neurons and input.ndim != 2:
            print("Input shape in BatchNorm not correct.")
            return input

        x_hat = (input - input.mean(dim=0)) * (input.var(dim=0, unbiased=False) + self.eps).pow(-0.5)
        out = self.gamma * x_hat + self.beta
        ########################
        # END OF YOUR CODE    #
        #######################

        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################

class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomBatchNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        inv_var = (input.var(dim=0, unbiased=False) + eps).pow(-0.5)
        xhat = (input - input.mean(dim=0)) * inv_var
        out = gamma * xhat + beta

        ctx.save_for_backward(xhat, gamma)
        ctx.inv_var = inv_var
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        xhat, gamma = ctx.saved_tensors
        grad_input = grad_gamma = grad_beta = None

        dxhat = grad_output * gamma  # element-wise multiplication
        batch_size = xhat.shape[0]

        if ctx.needs_input_grad[0] is True:
            inner_term = batch_size * dxhat - dxhat.sum(dim=0) - xhat * torch.sum(dxhat * xhat, dim=0)
            grad_input = ctx.inv_var / batch_size * inner_term

        if ctx.needs_input_grad[1] is True:
            grad_gamma = torch.sum(grad_output * xhat, dim=0)

        if ctx.needs_input_grad[2] is True:
            grad_beta = grad_output.sum(dim=0)
        ########################
        # END OF YOUR CODE    #
        #######################

        # return gradients of the three tensor inputs and None for the constant eps
        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

# noinspection PyUnresolvedReferences
class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.neurons = n_neurons
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if input.shape[1] != self.neurons and input.ndim != 2:
            print("Input shape in BatchNorm not correct.")
            return input

        custom_batch_norm = CustomBatchNormManualFunction()
        out = custom_batch_norm.apply(input, self.gamma, self.beta, self.eps)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
