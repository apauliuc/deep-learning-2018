"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2048, 1024, 1024, 512, 512, 256, 256'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 7500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 250

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    class_predicted = np.argmax(predictions, axis=1)
    class_target = np.argmax(targets, axis=1)
    accuracy = np.sum(class_predicted == class_target) / len(class_predicted)
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


# noinspection PyUnresolvedReferences
def train():
    """
    Performs training and evaluation of MLP model.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    opt = 'Adam'
    decay = 0

    # Import data
    cifar10_data = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_data = cifar10_data['train']
    test_data = cifar10_data['test']
    validation_data = cifar10_data['validation']

    input_size = np.prod(np.array([train_data.images[0].shape]))
    output_size = train_data.labels.shape[1]

    # Create model and optimizer
    model = MLP(input_size, dnn_hidden_units, output_size)
    criterion = nn.CrossEntropyLoss()
    if opt is 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=decay)
    elif opt is 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, weight_decay=decay)
    model.train()

    # Train & evaluate
    eval_loss = 0.0
    full_loss = []
    lossv = []
    accv = []

    for step in range(1, FLAGS.max_steps + 1):
        data, target = train_data.next_batch(FLAGS.batch_size)
        data, target = Variable(torch.from_numpy(data).float()), Variable(torch.from_numpy(target))

        optimizer.zero_grad()

        prediction = model(data)
        loss = criterion(prediction, torch.argmax(target, dim=1))
        loss.backward()
        optimizer.step()

        full_loss.append(loss.item())

        # Accuracy evaluation
        eval_loss += loss.item()
        if step % FLAGS.eval_freq == 0:
            model.eval()

            # test_x, test_y = test_data.next_batch(FLAGS.batch_size)
            test_x, test_y = test_data.images, test_data.labels
            test_x = Variable(torch.from_numpy(test_x).float())

            predicted_y = model(test_x)
            accuracy_result = accuracy(predicted_y.detach().numpy(), test_y)

            lossv.append(eval_loss / FLAGS.eval_freq)
            accv.append(accuracy_result)
            print('Step %d  -  accuracy: %.4f  -  loss: %.3f' % (step, accuracy_result, eval_loss / FLAGS.eval_freq))
            eval_loss = 0.0

            model.train()

    print("Training Done")

    # Save statistics for plotting
    # train_stats = {
    #     'loss': lossv,
    #     'acc': accv,
    #     'full_loss': full_loss
    # }
    # filename = 'loss_accuracy_files/mlp_torch_wnorm_' +\
    #            'lr_' + str(FLAGS.learning_rate) +\
    #            '_hidden_' + FLAGS.dnn_hidden_units +\
    #            '_steps_' + str(FLAGS.max_steps) +\
    #            '_optim_' + opt +\
    #            '_regularisation_' + str(decay) + '.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(train_stats, f)
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
