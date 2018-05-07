# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import numpy as np

from sagemaker_containers import env
import test


class Model(object):
    def __init__(self, weights=None, bias=1, loss=None, optimizer=None):
        self.epochs = None
        self.optimizer = optimizer
        self.loss = loss
        self.weights = weights
        self.bias = bias
        self.batch_size = None

    def fit(self, x, y, epochs=None, batch_size=None):
        self.weights = y / x + self.bias
        self.epochs = epochs
        self.batch_size = batch_size

    def save(self, model_dir):
        parameters = dict(weights=self.weights, bias=self.bias,
                          epochs=self.epochs, batch_size=self.batch_size)
        test.write_json(parameters, model_dir)

    @classmethod
    def load(cls, model_dir):
        return cls(**env.read_json(model_dir))

    def predict(self, data):
        return np.asarray(self.weights) * np.asarray(data)
