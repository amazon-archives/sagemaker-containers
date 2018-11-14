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

import contextlib
import importlib
import os
import sys
import tarfile

from mock import call, patch
import pytest
from six import PY2

from sagemaker_containers import _env, _errors, _modules, _params
import test

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


@patch('boto3.resource', autospec=True)
@pytest.mark.parametrize('url,bucket_name,key,dst',
                         [('S3://my-bucket/path/to/my-file', 'my-bucket', 'path/to/my-file', '/tmp/my-file'),
                          ('s3://my-bucket/my-file', 'my-bucket', 'my-file', '/tmp/my-file')])
def test_s3_download(resource, url, bucket_name, key, dst):
    region = 'us-west-2'
    os.environ[_params.REGION_NAME_ENV] = region

    _modules.s3_download(url, dst)

    chain = call('s3', region_name=region).Bucket(bucket_name).download_file(key, dst)
    assert resource.mock_calls == chain.call_list()


@patch.object(sys, 'path')
def test_prepare_module(path, entrypoint_type_module):
    _modules.prepare('/opt/ml/code', 'python_module')

    path.insert.assert_called_with(0, '/opt/ml/code')


@patch.object(sys, 'path')
def test_prepare_script(path, entrypoint_type_script):
    _modules.prepare('/opt/ml/code', 'train.py')

    path.insert.assert_called_with(0, '/opt/ml/code')


@patch('os.chmod')
@patch.object(sys, 'path')
def test_prepare_command(path, chmod, entrypoint_type_script):
    _modules.prepare('/opt/ml/code', 'train.sh')

    path.insert.assert_called_with(0, '/opt/ml/code')
    chmod.assert_called_with('/opt/ml/code/train.sh', 511)


def test_s3_download_wrong_scheme():
    with pytest.raises(ValueError, message="Expecting 's3' scheme, got: c in c://my-bucket/my-file"):
        _modules.s3_download('c://my-bucket/my-file', '/tmp/file')


@patch('sagemaker_containers._modules._check_error', autospec=True)
def test_install_module(check_error, entrypoint_type_module):
    path = 'c://sagemaker-pytorch-container'
    _modules.install(path, 'python_module.py')

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_error.assert_called_with(cmd, _errors.InstallModuleError, cwd=path)

    with patch('os.path.exists', return_value=True):
        _modules.install(path, 'python_module.py')

        check_error.assert_called_with(cmd + ['-r', 'requirements.txt'], _errors.InstallModuleError, cwd=path)


@patch('sagemaker_containers._modules._check_error', autospec=True)
def test_install_script(check_error, entrypoint_type_script, has_requirements):
    path = 'c://sagemaker-pytorch-container'
    _modules.install(path, 'train.py')

    with patch('os.path.exists', return_value=True):
        _modules.install(path, 'python_module.py')


@patch('sagemaker_containers._modules._check_error', autospec=True)
def test_install_fails(check_error, entrypoint_type_module):
    check_error.side_effect = _errors.ClientError()
    with pytest.raises(_errors.ClientError):
        _modules.install('git://aws/container-support', 'script')


@patch('sys.executable', None)
def test_install_no_python_executable(has_requirements, entrypoint_type_module):
    with pytest.raises(RuntimeError) as e:
        _modules.install('git://aws/container-support', 'train.py')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@contextlib.contextmanager
def patch_tmpdir():
    yield '/tmp'


@patch('importlib.import_module')
def test_exists(import_module):
    assert _modules.exists('my_module')

    import_module.side_effect = ImportError()

    assert not _modules.exists('my_module')


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_bash(log, popen, entrypoint_type_script):

    with pytest.raises(_errors.ExecuteUserScriptError):
        _modules.run('launcher.sh', ['--lr', '13'])

    cmd = ['/bin/sh', '-c', './launcher.sh --lr 13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_python(log, popen, entrypoint_type_script):

    with pytest.raises(_errors.ExecuteUserScriptError):
        _modules.run('launcher.py', ['--lr', '13'])

    cmd = [sys.executable, 'launcher.py', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_module(log, popen, entrypoint_type_module):

    with pytest.raises(_errors.ExecuteUserScriptError):
        _modules.run('module.py', ['--lr', '13'])

    cmd = [sys.executable, '-m', 'module', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('sagemaker_containers.training_env', lambda: {})
def test_run_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        _modules.run('wrong module')

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message


def test_python_executable_exception():
    with patch('sys.executable', None):
        with pytest.raises(RuntimeError):
            _modules.python_executable()


def test_run_module_wait():
    with patch('sagemaker_containers._modules.download_and_install') as download_and_install:
        with patch('sagemaker_containers._modules.run') as run:
            with pytest.warns(DeprecationWarning):
                _modules.run_module(uri='s3://url', args=['42'], cache=True)

                download_and_install.assert_called_with('s3://url', 'default_user_module_name', _env.code_dir)
                run.assert_called_with('default_user_module_name', ['42'], {}, True)


def test_run_module_no_wait():
    with patch('sagemaker_containers._modules.download_and_install') as download_and_install:
        with patch('sagemaker_containers._modules.run') as run:
            with pytest.warns(DeprecationWarning):
                _modules.run_module(uri='s3://url', args=['42'], cache=True, wait=False)

                download_and_install.assert_called_with('s3://url', 'default_user_module_name', _env.code_dir)
                run.assert_called_with('default_user_module_name', ['42'], {}, False)


@patch('os.makedirs')
@patch('shutil.copy2')
@patch('sagemaker_containers._modules.s3_download')
@patch('sagemaker_containers._modules.prepare')
@patch('sagemaker_containers._modules.install')
@patch('os.path.exists')
@pytest.mark.parametrize('exists', [False, True])
def test_download_and_install_local_file(path_exists, install, prepare, s3_download,
                                         copy2, makedirs, exists):
    path_exists.return_value = exists
    makedirs.return_value = exists

    _modules.download_and_install('/tmp/file', 'script', _env.code_dir)

    s3_download.assert_not_called()
    prepare.assert_called_with(_env.code_dir, 'script')
    install.assert_called_with(_env.code_dir, 'script')

    if not exists:
        makedirs.assert_called_with(_env.code_dir)
    copy2.assert_called_with('/tmp/file', os.path.join(_env.code_dir, 'script'))


@patch('os.makedirs')
@patch('shutil.rmtree')
@patch('shutil.copytree')
@patch('sagemaker_containers._modules.s3_download')
@patch('sagemaker_containers._modules.prepare')
@patch('sagemaker_containers._modules.install')
@patch('os.path.exists')
@pytest.mark.parametrize('exists', [False, True])
def test_download_and_install_local_folder(path_exists, install, prepare, s3_download, copytree,
                                           rmtree, makedirs, exists):
    path_exists.return_value = exists
    makedirs.return_value = exists

    _modules.download_and_install('/tmp', 'script', _env.code_dir)

    s3_download.assert_not_called()
    prepare.assert_called_with(_env.code_dir, 'script')
    install.assert_called_with(_env.code_dir, 'script')

    if exists:
        rmtree.assert_any_call(_env.code_dir)
    copytree.assert_called_with('/tmp', _env.code_dir)


@patch('shutil.copytree')
def test_download_and_install_local_directory(copytree):
    with patch('sagemaker_containers._modules.s3_download') as s3_download, \
            patch('sagemaker_containers._modules.prepare') as prepare, \
            patch('sagemaker_containers._modules.install') as install:
        _modules.download_and_install('/tmp', 'script', _env.code_dir)

        s3_download.assert_not_called()
        prepare.assert_called_with(_env.code_dir, 'script')
        install.assert_called_with(_env.code_dir, 'script')
        copytree.assert_called_with('/tmp', _env.code_dir)


class TestDownloadAndImport(test.TestBase):
    patches = [patch('sagemaker_containers._files.tmpdir', new=patch_tmpdir),
               patch('sagemaker_containers._modules.prepare', autospec=True),
               patch('sagemaker_containers._modules.install', autospec=True),
               patch('sagemaker_containers._modules.s3_download', autospec=True),
               patch('sagemaker_containers._modules.exists', autospec=True), patch('tarfile.open', autospec=True),
               patch('importlib.import_module', autospec=True), patch('six.moves.reload_module', autospec=True),
               patch('os.makedirs', autospec=True)]

    def test_without_cache(self):
        with tarfile.open() as tar_file:
            module = _modules.import_module('s3://bucket/my-module')

            assert module == importlib.import_module(_modules.DEFAULT_MODULE_NAME)

            _modules.s3_download.assert_called_with('s3://bucket/my-module', '/tmp/tar_file')

            tar_file.extractall.assert_called_with(path=_env.code_dir)
            _modules.prepare.assert_called_with(_env.code_dir, _modules.DEFAULT_MODULE_NAME)
            _modules.install.assert_called_with(_env.code_dir, _modules.DEFAULT_MODULE_NAME)

    def test_with_cache_and_module_already_installed(self):
        with tarfile.open() as tar_file:
            _modules.exists.return_value = True

            module = _modules.import_module('s3://bucket/my-module')

            assert module == importlib.import_module(_modules.DEFAULT_MODULE_NAME)

            _modules.s3_download.return_value.assert_not_called()
            os.makedirs.return_value.assert_not_called()

            tar_file.extractall.return_value.assert_not_called()
            _modules.prepare.return_value.assert_not_called()
            _modules.install.return_value.assert_not_called()

    def test_default_name(self):
        with tarfile.open() as tar_file:
            _modules.exists.return_value = False

            module = _modules.import_module('s3://bucket/my-module')

            assert module == importlib.import_module(_modules.DEFAULT_MODULE_NAME)

            _modules.s3_download.assert_called_with('s3://bucket/my-module', '/tmp/tar_file')
            os.makedirs.assert_called_with(_env.code_dir)

            tar_file.extractall.assert_called_with(path=_env.code_dir)
            _modules.prepare.assert_called_with(_env.code_dir, _modules.DEFAULT_MODULE_NAME)
            _modules.install.assert_called_with(_env.code_dir, _modules.DEFAULT_MODULE_NAME)

    def test_any_name(self):
        with tarfile.open() as tar_file:
            _modules.exists.return_value = False

            module = _modules.import_module('s3://bucket/my-module', 'another_module_name')

            assert module == importlib.import_module('another_module_name')

            _modules.s3_download.assert_called_with('s3://bucket/my-module', '/tmp/tar_file')
            os.makedirs.assert_called_with(_env.code_dir)

            tar_file.extractall.assert_called_with(path=_env.code_dir)
            _modules.prepare.assert_called_with(_env.code_dir, 'another_module_name')
            _modules.install.assert_called_with(_env.code_dir, 'another_module_name')
