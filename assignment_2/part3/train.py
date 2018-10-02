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

import os
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from part3.dataset import TextDataset
from part3.model import TextGenerationModel


################################################################################
def save_checkpoint(save_dict, path='', filename='checkpoint.pth.tar'):
    torch.save(save_dict, path + filename)


def train(config):
    creation_time = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')
    save_dir = config.summary_path + creation_time + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + 'params.txt', 'w+', encoding='utf-8') as f:
        f.write(str(config))

    summary_file_path = save_dir + 'generated_text.txt'
    summary_file = open(summary_file_path, 'w+', encoding='utf-8')

    best_accuracy = 0.0

    # Initialize the device which to run the model on
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        l_type = torch.cuda.LongTensor
    else:
        device = torch.device('cpu')
        l_type = torch.LongTensor

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file, seq_length=config.seq_length+1, batch_size=config.batch_size,
                          train_steps=config.train_steps)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    # Initialize the model that we are going to use
    model = TextGenerationModel(vocabulary_size=dataset.vocab_size, lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers, dropout_prob=1-config.dropout_keep_prob,
                                temperature=config.temperature, device=device)

    # Setup the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        criterion.cuda()
        model.cuda()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = Variable(torch.stack(batch_inputs).transpose(0, 1)).type(l_type)
        batch_targets = Variable(torch.stack(batch_targets).transpose(0, 1)).type(l_type)

        optimizer.zero_grad()

        batch_predicted, _, _ = model(batch_inputs)
        loss = criterion(batch_predicted.transpose(1, 2), batch_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        loss = loss.item()
        accuracy = float(torch.sum(torch.argmax(batch_predicted, dim=2) == batch_targets).item()) \
            / (config.seq_length * config.batch_size)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            if step > 50:
                model.cpu()
                save_checkpoint({
                    'step': step + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy,
                    'optimizer': optimizer.state_dict(),
                }, path=save_dir, filename='best_checkpoint.pth.tar')
                if torch.cuda.is_available():
                    model.cuda()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        # Print statistics
        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04.0f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        # Sample text
        if step % config.sample_every == 0 and step != 0:
            model.eval()
            with torch.no_grad():
                sequence = Variable(torch.empty(1, 1)).type(l_type)
                sequence.random_(to=dataset.vocab_size)

                for t in range(0, config.seq_length):
                    out_seq, _, _ = model(sequence)
                    next_char = torch.tensor([[torch.argmax(out_seq, dim=2)[0, t]]]).type(l_type)
                    sequence = torch.cat((sequence, next_char), dim=1)

                generated_sequence = dataset.convert_to_string(sequence.detach().cpu().numpy().squeeze())
                write_to_file = "{}\n\n".format(generated_sequence)
                summary_file.write(write_to_file)
                summary_file.flush()

            model.train()

        # Decay learning rate at periodic intervals
        if step % config.learning_rate_step == 0 and step != 0:
            new_lr = config.learning_rate * (config.learning_rate_decay ** (step // config.learning_rate_step))
            print("Learning rate decreased to {}".format(new_lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Save model checkpoint every 10,000 steps
        if step % 10000 == 0 and step != 0:
            model.cpu()
            save_checkpoint({
                'step': step + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, path=save_dir, filename='checkpoint_step_{}.pth.tar'.format(step))
            if torch.cuda.is_available():
                model.cuda()

        # Stop training on max training steps
        if step == config.train_steps or loss <= 1:
            print("Reached loss {:.5f} at step {:04d}".format(loss, step))
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, help="Path to a .txt file to train on",
                        default="rsc/book_EN_grimms_fairy_tails.txt")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=50000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=5000, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temperature', type=float, default=1, help="Temperature value for division of outputs")

    config = parser.parse_args()

    # Train the model
    train(config)
