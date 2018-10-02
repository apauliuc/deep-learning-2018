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

import argparse
import time
import os
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM


# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

# noinspection PyShadowingNames
def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    if torch.cuda.is_available():
        device = torch.device('cuda')
        l_type = torch.cuda.LongTensor
        f_type = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        l_type = torch.LongTensor
        f_type = torch.FloatTensor

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                           config.batch_size, device)
    else:
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                     config.batch_size, device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        criterion.cuda()
        model.cuda()

    losses = []
    accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        model.zero_grad()
        optimizer.zero_grad()

        batch_inputs = batch_inputs.type(f_type)
        batch_targets = batch_targets.type(l_type)

        batch_predicted = model(batch_inputs)
        loss = criterion(batch_predicted, batch_targets)
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()

        loss = loss.item()
        correct_predicted = torch.sum(torch.argmax(batch_predicted, dim=1) == batch_targets).item()
        accuracy = float(correct_predicted) / config.batch_size

        losses.append(loss)
        accuracies.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % 100 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.4f}, Loss = {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if loss <= 0.001:
            print('Model converged in {:04d} steps at loss {}'.format(step, loss))
            break

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    # save_dict = {
    #     'loss': losses,
    #     'accuracy': accuracies,
    #     'summary': str(config)
    # }
    # filename = '/home/lgpu0080/results_part1/history_' + str(config.model_type) +\
    #            '_length_' + str(config.input_length) + '.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(save_dict, f)


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    print(config)

    # Train the model
    train(config)
