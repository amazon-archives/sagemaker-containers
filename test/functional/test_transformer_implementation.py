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
from __future__ import absolute_import

import os

import numpy as np

from sagemaker_containers import content_types, env, serializers, status_codes, transformers, worker
import test
from test import miniml


class MiniMlTransformer(transformers.BaseTransformer):
    def predict_fn(self, model, data):
        return model.predict(data)

    def model_fn(self, model_dir):
        return miniml.Model.load(os.path.join(model_dir, 'minimlmodel'))


def test_transformer_implementation():
    test.create_resource_config()
    test.create_input_data_config()
    test.create_hyperparameters_config({'sagemaker_program': 'user_script.py'})

    model_path = os.path.join(env.TrainingEnv().model_dir, 'minimlmodel')
    miniml.Model(W=[6, 9, 42]).save(model_path)

    transformer = MiniMlTransformer()
    transformer.initialize()

    with worker.run(transformer.transform, transformer.initialize, module_name='miniml').test_client() as client:
        payload = [6, 9, 42.]
        response = post(client, payload, content_types.JSON)

        assert response.status_code == status_codes.OK
        assert response.get_data().decode('utf-8') == '[36.0, 81.0, 1764.0]'

        response = post(client, payload, content_types.CSV)

        assert response.status_code == status_codes.OK
        assert response.get_data().decode('utf-8') == '36.0\n81.0\n1764.0\n'

        response = post(client, payload, content_types.NPY)

        assert response.status_code == status_codes.OK
        response_data = serializers.loads(response.get_data(), content_types.NPY)

        np.testing.assert_array_almost_equal(response_data, np.asarray([36., 81., 1764.]))


def post(client, payload, content_type):
    return client.post(path='/invocations', headers={'accept': content_type},
                       data=serializers.dumps(payload, content_type), content_type=content_type)
