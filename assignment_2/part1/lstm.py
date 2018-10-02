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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.device = device
        self.num_hidden = num_hidden
        self.batch_size = batch_size

        # Parameter initialisation
        self.W_gx = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, input_dim)))
        self.W_gh = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.W_ix = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, input_dim)))
        self.W_ih = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.W_fx = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, input_dim)))
        self.W_fh = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.W_ox = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, input_dim)))
        self.W_oh = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.W_ph = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_classes, num_hidden)))

        self.b_g = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_i = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_f = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_o = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        self.h = None
        self.c = None

    def forward(self, x):
        self.h = torch.zeros(self.num_hidden, self.batch_size).to(device=self.device)
        self.c = torch.zeros(self.num_hidden, self.batch_size).to(device=self.device)

        for t in range(self.seq_length):
            x_t = x[:, t].reshape(1, -1).to(device=self.device)

            g = torch.tanh(self.W_gx @ x_t + self.W_gh @ self.h + self.b_g).to(device=self.device)
            i = torch.sigmoid(self.W_ix @ x_t + self.W_ih @ self.h + self.b_i).to(device=self.device)
            f = torch.sigmoid(self.W_fx @ x_t + self.W_fh @ self.h + self.b_f).to(device=self.device)
            o = torch.sigmoid(self.W_ox @ x_t + self.W_oh @ self.h + self.b_o).to(device=self.device)
            self.c = g * i + self.c * f
            self.h = torch.tanh(self.c) * o

        return torch.t(self.W_ph @ self.h + self.b_p)
