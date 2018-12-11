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

logging.basicConfig(level=logging.INFO)

from sagemaker.tensorflow import TensorFlow

dir_path = os.path.realpath(__file__)


def test_mpi():
    resourcers = os.path.realpath(
        os.path.join(dir_path, '..', '..', 'resources', 'openmpi', 'launcher.sh'))
    est = TensorFlow(entry_point=resourcers,
                     image_name='openmpi',
                     role='SageMakerRole',
                     train_instance_count=2,
                     framework_version='1.11',
                     py_version='py3',
                     train_instance_type='local',
                     hyperparameters={'sagemaker_mpi_enabled': True,
                                      'sagemaker_network_interface_name': 'eth0'},
                     distributions={'mpi': {'enabled': True}})

    est.fit()
