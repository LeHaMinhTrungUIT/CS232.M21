# Phần này để Train và lưu model

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import pandas as pd

batch_size = 1000
image_size = 160


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=666)

            df = pd.read_csv('features.csv')

            labels = df.iloc[:, -1].values

            emb_array = df.iloc[:, 0:-1].values
            classifier_filename_exp = os.path.expanduser(
                "./model/facemodels.pkl")

            sav = pd.read_csv('labels.csv', header=None)
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = sav.iloc[:, 0].values

            print("class_names: ", class_names)

            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' %
                  classifier_filename_exp)


main()
