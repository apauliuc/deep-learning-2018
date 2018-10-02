# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, dropout_prob=0, temperature=1,
                 device='cpu'):
        super(TextGenerationModel, self).__init__()
        self.vocab_size = vocabulary_size
        self.device = device
        self.temperature = temperature

        self.lstm = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers,
                            batch_first=True, dropout=dropout_prob).to(device=self.device)
        self.linear = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size).to(device=self.device)

    def forward(self, x, hc_0=None):
        one_hot = torch.zeros((*x.shape, self.vocab_size)).to(device=self.device)
        x = x.unsqueeze(-1)
        one_hot.scatter_(2, x, 1)

        if hc_0 is not None:
            lstm_out, (h, c) = self.lstm(one_hot, hc_0)
        else:
            lstm_out, (h, c) = self.lstm(one_hot)
        out = torch.log_softmax(self.linear(lstm_out) / self.temperature, dim=2)
        return out, h, c
