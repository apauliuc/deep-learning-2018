################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device=torch.device('cpu')):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.device = device
        self.num_hidden = num_hidden
        self.batch_size = batch_size

        # Parameter initialisation
        self.W_hx = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, input_dim)))
        self.W_hh = nn.Parameter(nn.init.xavier_normal_(torch.empty((num_hidden, num_hidden))))
        self.W_ph = nn.Parameter(nn.init.xavier_normal_(torch.empty((num_classes, num_hidden))))

        self.b_h = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

    def forward(self, x):
        h = torch.zeros(self.num_hidden, self.batch_size).to(device=self.device)

        for t in range(self.seq_length):
            x_t = x[:, t].reshape(1, -1).to(device=self.device)
            h = torch.tanh(self.W_hx @ x_t + self.W_hh @ h + self.b_h)

        return torch.t(self.W_ph @ h + self.b_p)
