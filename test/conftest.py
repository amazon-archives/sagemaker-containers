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
import logging
import os

from mock import patch
import pytest
import six

import sagemaker_containers.environment as environment

logger = logging.getLogger(__name__)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


@pytest.fixture(name='base_path')
def fixture_base_path(tmpdir):
    yield str(tmpdir)


@pytest.fixture
def create_base_path(base_path):

    with patch.dict('os.environ', {'BASE_PATH': base_path}):
        six.moves.reload_module(environment)
        os.makedirs(environment.MODEL_PATH)
        os.makedirs(environment.INPUT_DATA_CONFIG_PATH)
        os.makedirs(environment.OUTPUT_DATA_PATH)

        yield base_path
    six.moves.reload_module(environment)
