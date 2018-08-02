# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import os
import pickle

from sagemaker_containers.beta.framework import transformer, worker


def predict_fn(data, model):
    return model.predict(data)


def model_fn(model_dir):
    with open(os.path.join(model_dir, 'decision-tree-model.pkl'), 'r') as inp:
        return pickle.load(inp)


def main(environ, start_response):
    user_module_transformer = transformer.Transformer(model_fn=model_fn, predict_fn=predict_fn)

    user_module_transformer.initialize()

    app = worker.Worker(transform_fn=user_module_transformer.transform, module_name='scikit-server')
    return app(environ, start_response)
