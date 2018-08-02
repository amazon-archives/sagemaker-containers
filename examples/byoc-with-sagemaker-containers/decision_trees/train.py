# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import argparse
import logging
import os
import pickle
import sys

import pandas as pd
from sklearn import tree

logging.basicConfig(stream=sys.stdout)


# The function to execute the training.
def main(data_dir, model_dir, max_leaf_nodes):
    print('Starting the training.')

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(data_dir))

    raw_data = [pd.read_csv(file, header=None) for file in input_files]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.ix[:, 0]
    train_X = train_data.ix[:, 1:]

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(train_X, train_y)

    # save the model
    with open(os.path.join(model_dir, 'decision-tree-model.pkl'), 'w') as out:
        pickle.dump(clf, out)
    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--max-leaf-nodes', type=int)
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args = parser.parse_args()

    main(args.data_dir, args.model_dir, args.max_leaf_nodes)
