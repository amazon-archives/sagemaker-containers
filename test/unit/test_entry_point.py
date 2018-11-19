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
import sys

from mock import patch
import pytest
from six import PY2

from sagemaker_containers import _env, _errors, entrypoint

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@pytest.fixture
def entrypoint_type_module():
    with patch('os.listdir', lambda x: ('setup.py',)):
        yield


@pytest.fixture(autouse=True)
def entrypoint_type_script():
    with patch('os.listdir', lambda x: ()):
        yield


@pytest.fixture()
def has_requirements():
    with patch('os.path.exists', lambda x: x.endswith('requirements.txt')):
        yield


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_module(check_error, entrypoint_type_module):
    path = 'c://sagemaker-pytorch-container'
    entrypoint.install('python_module.py', path)

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_error.assert_called_with(cmd, _errors.InstallModuleError, cwd=path)

    with patch('os.path.exists', return_value=True):
        entrypoint.install('python_module.py', path)

        check_error.assert_called_with(cmd + ['-r', 'requirements.txt'], _errors.InstallModuleError, cwd=path)


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_script(check_error, entrypoint_type_module, has_requirements):
    path = 'c://sagemaker-pytorch-container'
    entrypoint.install('train.py', path)

    with patch('os.path.exists', return_value=True):
        entrypoint.install(path, 'python_module.py')


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_fails(check_error, entrypoint_type_module):
    check_error.side_effect = _errors.ClientError()
    with pytest.raises(_errors.ClientError):
        entrypoint.install('git://aws/container-support', 'script')


@patch('sys.executable', None)
def test_install_no_python_executable(has_requirements, entrypoint_type_module):
    with pytest.raises(RuntimeError) as e:
        entrypoint.install('train.py', 'git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_bash(log, popen, entrypoint_type_script):
    with pytest.raises(_errors.ExecuteUserScriptError):
        entrypoint.call('launcher.sh', ['--lr', '13'])

    cmd = ['/bin/sh', '-c', './launcher.sh --lr 13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_python(log, popen, entrypoint_type_script):
    with pytest.raises(_errors.ExecuteUserScriptError):
        entrypoint.call('launcher.py', ['--lr', '13'])

    cmd = [sys.executable, 'launcher.py', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_module(log, popen, entrypoint_type_module):
    with pytest.raises(_errors.ExecuteUserScriptError):
        entrypoint.call('module.py', ['--lr', '13'])

    cmd = [sys.executable, '-m', 'module', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('sagemaker_containers.training_env', lambda: {})
def test_run_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        entrypoint.call('wrong module')

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers.entrypoint.call')
@patch('os.chmod')
def test_run_module_wait(chmod, call, download_and_extract):
    entrypoint.run(uri='s3://url', user_entry_point='launcher.sh', args=['42'])

    download_and_extract.assert_called_with('s3://url', 'launcher.sh', _env.code_dir)
    call.assert_called_with('launcher.sh', ['42'], {}, True)
    chmod.assert_called_with(os.path.join(_env.code_dir, 'launcher.sh'), 511)


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers.entrypoint.call')
def test_run_module_no_wait(call, download_and_extract, entrypoint_type_module):
    with pytest.raises(_errors.InstallModuleError):
        entrypoint.run(uri='s3://url', user_entry_point='default_user_module_name', args=['42'], wait=False)

        download_and_extract.assert_called_with('s3://url', 'default_user_module_name', _env.code_dir)
        call.assert_called_with('default_user_module_name', ['42'], {}, False)
