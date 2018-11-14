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

import enum
import importlib
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import warnings

import boto3
import six
from six.moves.urllib.parse import urlparse

from sagemaker_containers import _env, _errors, _files, _logging, _params

logger = _logging.get_logger()

DEFAULT_MODULE_NAME = 'default_user_module_name'


def s3_download(url, dst):  # type: (str, str) -> None
    """Download a file from S3.

    Args:
        url (str): the s3 url of the file.
        dst (str): the destination where the file will be saved.
    """
    url = urlparse(url)

    if url.scheme != 's3':
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    bucket, key = url.netloc, url.path.lstrip('/')

    region = os.environ.get('AWS_REGION', os.environ.get(_params.REGION_NAME_ENV))
    s3 = boto3.resource('s3', region_name=region)

    s3.Bucket(bucket).download_file(key, dst)


def prepare(path, name):  # type: (str, str) -> None
    """Prepare the user code entry_point to be executed as follow:
        - add the path to sys path
        - if the entrypoint is a command, gives exec permissions to the script

    Args:
        path (str): path to directory with the script or module.
        name (str): name of the script or module.
    """
    if path not in sys.path:
        sys.path.insert(0, path)

    if _entry_point_type(path, name) is EntryPointType.COMMAND:
        os.chmod(os.path.join(path, name), 511)


def install(path, name):  # type: (str, str) -> None
    """Install Python module and dependencies.
       If the entry point root folder:
        - has a setup.py, it installs as python package.
        - has a requirements.txt file, it installs its dependencies.

    Args:
        path (str):  Real path location of the Python module.
    """
    entry_point_type = _entry_point_type(path, name)

    if entry_point_type is EntryPointType.PYTHON_PACKAGE:
        cmd = '%s -m pip install -U . ' % python_executable()

        if os.path.exists(os.path.join(path, 'requirements.txt')):
            cmd += '-r requirements.txt'

        logger.info('Installing module with the following command:\n%s', cmd)

        _check_error(shlex.split(cmd), _errors.InstallModuleError, cwd=path)


def exists(name):  # type: (str) -> bool
    """Return True if the module exists. Return False otherwise.

    Args:
        name (str): module name.

    Returns:
        (bool): boolean indicating if the module exists or not.
    """
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    else:
        return True


def download_and_install(uri, name, path):  # type: (str, str, str) -> None
    """Download, prepare and install a compressed tar file from S3 or local directory as an entry point.

    SageMaker Python SDK saves the user provided entry points as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.

    Args:
        name (str): name of the entry point.
        uri (str): the location of the entry point.
        path (bool): The path where the script will be installed. It will not download and install the
                        if the path already has the user entry point.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.listdir(path):
        with _files.tmpdir() as tmpdir:
            if uri.startswith('s3://'):
                dst = os.path.join(tmpdir, 'tar_file')
                s3_download(uri, dst)

                with tarfile.open(name=dst, mode='r:gz') as t:
                    t.extractall(path=path)

            elif os.path.isdir(uri):
                if os.path.exists(path):
                    shutil.rmtree(path)
                shutil.copytree(uri, path)
            else:
                shutil.copy2(uri, os.path.join(path, name))

    prepare(path, name)

    install(path, name)


def run(module_name, args=None, env_vars=None, wait=True):  # type: (str, list, dict, bool) -> Popen
    """Runs the entry-point, passing env_vars as environment variables and args as command arguments.
    If the entry point is:
        - A Python package: executes the packages as >>> env_vars python -m module_name + args
        - A Python script: executes the script as >>> env_vars python module_name + args
        - Any other: executes the command as >>> env_vars /bin/sh -c ./module_name + args

    Example:

        >>>import sagemaker_containers
        >>>from sagemaker_containers.beta.framework import mapping, modules

        >>>env = sagemaker_containers.training_env()
        {'channel-input-dirs': {'training': '/opt/ml/input/training'}, 'model_dir': '/opt/ml/model', ...}


        >>>hyperparameters = env.hyperparameters
        {'batch-size': 128, 'model_dir': '/opt/ml/model'}

        >>>args = mapping.to_cmd_args(hyperparameters)
        ['--batch-size', '128', '--model_dir', '/opt/ml/model']

        >>>env_vars = mapping.to_env_vars()
        ['SAGEMAKER_CHANNELS':'training', 'SAGEMAKER_CHANNEL_TRAINING':'/opt/ml/input/training',
        'MODEL_DIR':'/opt/ml/model', ...}

        >>>modules.run('user_script', args, env_vars)
        SAGEMAKER_CHANNELS=training SAGEMAKER_CHANNEL_TRAINING=/opt/ml/input/training \
        SAGEMAKER_MODEL_DIR=/opt/ml/model python -m user_script --batch-size 128 --model_dir /opt/ml/model

    Args:
        module_name (str): module name in the same format required by python -m <module-name> cli command.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
    """
    args = args or []
    env_vars = env_vars or {}

    # TODO (mvsusp): module_name will be deprecated to entry_point_name in a follow up pr that will refactor _modules
    entry_point_type = _entry_point_type(_env.code_dir, module_name)

    if entry_point_type is EntryPointType.PYTHON_PACKAGE:
        cmd = [python_executable(), '-m', module_name.replace('.py', '')] + args
    elif entry_point_type is EntryPointType.PYTHON_PROGRAM:
        cmd = [python_executable(), module_name] + args
    else:
        cmd = ['/bin/sh', '-c', './%s %s' % (module_name, ' '.join(args))]

    _logging.log_script_invocation(cmd, env_vars)

    if wait:
        return _check_error(cmd, _errors.ExecuteUserScriptError)

    else:
        return _make_process(cmd, _errors.ExecuteUserScriptError)


def _make_process(cmd, error_class, cwd=None, **kwargs):
    try:
        return subprocess.Popen(cmd, env=os.environ, cwd=cwd or _env.code_dir, **kwargs)
    except Exception as e:
        six.reraise(error_class, error_class(e), sys.exc_info()[2])


def _check_error(cmd, error_class, **kwargs):
    process = _make_process(cmd, error_class, **kwargs)
    return_code = process.wait()

    if return_code:
        raise error_class(return_code=return_code, cmd=' '.join(cmd))
    return process


def python_executable():
    """Returns the real path for the Python executable, if it exists. Returns RuntimeError otherwise.

    Returns:
        (str): the real path of the current Python executable
    """
    if not sys.executable:
        raise RuntimeError('Failed to retrieve the real path for the Python executable binary')
    return sys.executable


def import_module(uri, name=DEFAULT_MODULE_NAME, cache=None):  # type: (str, str, bool) -> module
    """Download, prepare and install a compressed tar file from S3 or provided directory as a module.
    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, if provided, and transforms it as a module, and installs it.
    Args:
        name (str): name of the script or module.
        uri (str): the location of the module.
        cache (bool): default True. It will not download and install the module again if it is already installed.
    Returns:
        (module): the imported module
    """
    _warning_cache_deprecation(cache)

    name = name.replace('.py', '')
    download_and_install(uri, name, _env.code_dir)

    try:
        module = importlib.import_module(name)
        six.moves.reload_module(module)

        return module
    except Exception as e:
        six.reraise(_errors.ImportModuleError, _errors.ImportModuleError(e), sys.exc_info()[2])


def run_module(uri, args, env_vars=None, name=DEFAULT_MODULE_NAME, cache=None, wait=True):
    # type: (str, list, dict, str, bool, bool) -> Popen
    """Download, prepare and executes a compressed tar file from S3 or provided directory as an entry point.

    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, transforms it as a module, and executes it.
    Args:
        uri (str): the location of the entry point.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        name (str): name of the user entry point.
        cache (bool): If True it will avoid downloading the module again, if already installed.
        wait (bool): If True run_module will wait for the user module to exit and check the exit code,
                     otherwise it will launch the user module with subprocess and return the process object.
    """
    _warning_cache_deprecation(cache)
    env_vars = env_vars or {}
    env_vars = env_vars.copy()

    download_and_install(uri, name, _env.code_dir)

    write_env_vars(env_vars)

    return run(name, args, env_vars, wait)


def write_env_vars(env_vars=None):  # type: (dict) -> None
    """Write the dictionary env_vars in the system, as environment variables.

    Args:
        env_vars ():

    Returns:

    """
    env_vars = env_vars or {}
    env_vars['PYTHONPATH'] = ':'.join(sys.path)

    for name, value in env_vars.items():
        os.environ[name] = value


class EntryPointType(enum.Enum):
    PYTHON_PACKAGE = 'PYTHON_PACKAGE'
    PYTHON_PROGRAM = 'PYTHON_PROGRAM'
    COMMAND = 'COMMAND'


def _entry_point_type(path, name):  # type: (str, str) -> EntryPointType
    if 'setup.py' in os.listdir(path):
        return EntryPointType.PYTHON_PACKAGE
    elif name.endswith('.py'):
        return EntryPointType.PYTHON_PROGRAM
    else:
        return EntryPointType.COMMAND


def _has_requirements(path):  # type: (str) -> None
    return os.path.exists(os.path.join(path, 'requirements.txt'))


def _install_requirements(path):  # type: (str) -> None
    if _has_requirements(path):
        cmd = '%s -m pip install -r requirements.txt' % python_executable()
        logger.info('Installing requirements.txt with the following command:\n%s', cmd)
        _check_error(shlex.split(cmd), _errors.InstallModuleError, cwd=path)


def _warning_cache_deprecation(cache):
    if cache is not None:
        msg = 'the cache parameter is unnecessary anymore. Cache is always set to True'
        warnings.warn(msg, DeprecationWarning)
