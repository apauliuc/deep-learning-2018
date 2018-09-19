"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    cifar10_data = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_data = cifar10_data['train']
    test_data = cifar10_data['test']

    test_batch_size = 2000
    num_steps = test_data.num_examples / test_batch_size

    input_channels = train_data.images[0].shape[0]
    output_size = train_data.labels.shape[1]

    # Create model and optimizer
    model = ConvNet(input_channels, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    model.train()

    # Train & evaluate
    eval_loss = 0.0
    all_losses = []
    lossv = []
    train_accv = []
    test_accv = []

    for step in np.arange(1, FLAGS.max_steps + 1):
        data, target = train_data.next_batch(FLAGS.batch_size)
        data, target = Variable(torch.from_numpy(data).float()), Variable(torch.from_numpy(target))

        optimizer.zero_grad()

        prediction = model(data)
        loss = criterion(prediction, torch.argmax(target, dim=1))
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())

        # Accuracy evaluation
        eval_loss += loss.item()
        if step % FLAGS.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                predicted_test = None

                for _ in np.arange(num_steps):
                    test_x, test_y = test_data.next_batch(test_batch_size)
                    test_x = Variable(torch.from_numpy(test_x).float())

                    predicted_y = model(test_x)
                    predicted_y = predicted_y.detach().numpy()

                    if predicted_test is None:
                        predicted_test = predicted_y
                    else:
                        predicted_test = np.vstack((predicted_test, predicted_y))

                accuracy_result = accuracy(predicted_test, test_data.labels)

                lossv.append(eval_loss / FLAGS.eval_freq)
                train_accv.append(accuracy(prediction.detach().numpy(), target.detach().numpy()))
                test_accv.append(accuracy_result)
                print('-- Step %d  -  accuracy: %.3f  -  loss: %.3f'
                      % (step, accuracy_result, eval_loss / FLAGS.eval_freq))
                eval_loss = 0.0

                model.train()

    print("Training Done")

    # Save statistics for plotting
    # train_stats = {
    #     'all_loss': all_losses,
    #     'eval_loss': lossv,
    #     'test_acc': test_accv,
    #     'train_acc': train_accv
    # }
    # with open('loss_accuracy_files/convnet_torch.pkl', 'wb') as f:
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
