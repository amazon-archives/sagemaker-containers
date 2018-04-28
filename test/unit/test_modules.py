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
from __future__ import absolute_import

import os
import subprocess
import sys

from mock import call, MagicMock, mock_open, patch
import pytest
from six import PY2

import sagemaker_containers as smc
from test.mocks import assert_called_with, patch_context_manager, patch_with_validation

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('boto3.resource', autospec=True)
@pytest.mark.parametrize('url,bucket_name,key,dst', [
    ('S3://my-bucket/path/to/my-file', 'my-bucket', 'path/to/my-file', '/tmp/my-file'),
    ('s3://my-bucket/my-file', 'my-bucket', 'my-file', '/tmp/my-file')
])
def test_s3_download(resource, url, bucket_name, key, dst):
    smc.modules.s3_download(url, dst)

    chain = call('s3').Bucket(bucket_name).download_file(key, dst)
    assert resource.mock_calls == chain.call_list()


@patch(builtins_open, mock_open())
@patch('os.path.exists', lambda x: False)
def test_prepare():
    smc.modules.prepare('c:/path/to/', 'my-module')
    open.assert_called_with('c:/path/to/setup.py', 'w')

    content = os.linesep.join(['from setuptools import setup',
                               'setup(name="my-module", py_modules=["my-module"])'])

    open().write.assert_called_with(content)


@patch(builtins_open, mock_open())
@patch('os.path.exists', lambda x: True)
def test_prepare_already_prepared():
    smc.modules.prepare('c:/path/to/', 'my-module')
    open.assert_not_called()


def test_s3_download_wrong_scheme():
    with pytest.raises(ValueError, message="Expecting 's3' scheme, got: c in c://my-bucket/my-file"):
        smc.modules.s3_download('c://my-bucket/my-file', '/tmp/file')


@patch('subprocess.check_call', autospec=True)
def test_install(check_call):
    smc.modules.install('c://sagemaker-pytorch-container')

    check_call.assert_called_with([sys.executable, '-m', 'pip', 'install', 'c://sagemaker-pytorch-container', '-U'])


@patch('subprocess.check_call')
def test_install_fails(check_call):
    check_call.side_effect = subprocess.CalledProcessError(1, 'returned non-zero exit status 1')
    with pytest.raises(RuntimeError) as e:
        smc.modules.install('git://aws/container-support')
    assert str(e.value).startswith('Failed to pip install git://aws/container-support:')


@patch('sys.executable', None)
def test_install_no_python_executable():
    with pytest.raises(RuntimeError) as e:
        smc.modules.install('git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@patch_with_validation('importlib.import_module', smc.modules.DEFAULT_MODULE_NAME)
@patch_with_validation('os.makedirs', '/tmp/module_dir')
@patch_context_manager('sagemaker_containers.environment.temporary_directory', return_value='/tmp')
@patch_with_validation('sagemaker_containers.modules.prepare', '/tmp/module_dir', smc.modules.DEFAULT_MODULE_NAME)
@patch_with_validation('sagemaker_containers.modules.install', '/tmp/module_dir')
@patch_with_validation('sagemaker_containers.modules.s3_download', 's3://bucket/my-module', '/tmp/tar_file')
@patch('tarfile.open')
def test_s3_download_import_default_name(tarfile_open):
    smc.modules.download_and_import('s3://bucket/my-module')

    with assert_called_with(tarfile_open, name='/tmp/tar_file', mode='r:gz') as t:
        t.extractall.assert_called_with(path='/tmp/module_dir')


@patch_with_validation('importlib.import_module', 'another_module_name')
@patch_with_validation('os.makedirs', '/tmp/module_dir')
@patch_context_manager('sagemaker_containers.environment.temporary_directory', return_value='/tmp')
@patch_with_validation('sagemaker_containers.modules.prepare', '/tmp/module_dir', 'another_module_name')
@patch_with_validation('sagemaker_containers.modules.install', '/tmp/module_dir')
@patch_with_validation('sagemaker_containers.modules.s3_download', 's3://bucket/my-module', '/tmp/tar_file')
@patch('tarfile.open')
def test_s3_download_import(tarfile_open):
    smc.modules.download_and_import('s3://bucket/my-module', 'another_module_name')

    with assert_called_with(tarfile_open, name='/tmp/tar_file', mode='r:gz') as t:
        t.extractall.assert_called_with(path='/tmp/module_dir')


@patch('sagemaker_containers.modules.prepare', MagicMock(side_effect=ValueError('nothing to open')))
@patch('sagemaker_containers.modules.s3_download', MagicMock)
@patch('tempfile.NamedTemporaryFile', MagicMock)
@patch_with_validation('os.makedirs', '/tmp/module_dir')
@patch_context_manager('sagemaker_containers.environment.temporary_directory', return_value='/tmp')
@patch(builtins_open, mock_open())
@patch('tarfile.open', MagicMock)
def test_s3_download_and_import_deletes_tmp_dir():
    with pytest.raises(ValueError):
        smc.modules.download_and_import('s3://bucket/my-module', 'another_module_name')
