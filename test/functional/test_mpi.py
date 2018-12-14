# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import logging
import os

import pytest
from sagemaker.tensorflow import TensorFlow

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize('py_version', ['py2', 'py3'])
def test_mpi(py_version, tmpdir):
    dir_path = os.path.realpath(__file__)
    source_dir = os.path.realpath(os.path.join(dir_path, '..', '..', 'resources', 'openmpi'))

    estimator = TensorFlow(entry_point='launcher.sh',
                           image_name='openmpi',
                           role='SageMakerRole',
                           train_instance_count=2,
                           framework_version='1.11',
                           py_version=py_version,
                           source_dir=source_dir,
                           train_instance_type='local',
                           hyperparameters={
                               'sagemaker_mpi_enabled': True,
                               'sagemaker_mpi_custom_mpi_options': '-verbose',
                               'sagemaker_network_interface_name': 'eth0'
                           })

    estimator.fit()
