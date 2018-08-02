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
import errno
import os

from mock import Mock, patch, PropertyMock

import sagemaker_containers
from sagemaker_containers import _errors, _trainer


class TrainingEnv(Mock):
    framework_module = 'my_framework:entry_point'
    log_level = 20


@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
def test_train(import_module):
    framework = Mock()
    import_module.return_value = framework
    _trainer.train()

    import_module.assert_called_with('my_framework')
    framework.entry_point.assert_called()


@patch('sagemaker_containers._files.write_success_file')
@patch('sagemaker_containers._trainer._exit_processes')
@patch('sagemaker_containers._logging.configure_logger')
@patch('sagemaker_containers._modules.run')
@patch('sagemaker_containers._modules.write_env_vars')
@patch('sagemaker_containers._env.TrainingEnv.framework_module', PropertyMock(return_value='my_framework'))
def test_train_script_mode(write_env_vars, run, configure_logger,
                           _exit_processes, write_success_file):
    _trainer.train()

    env = sagemaker_containers.training_env()
    env_vars = env.to_env_vars()

    configure_logger.assert_called_with(env.log_level)
    write_env_vars.assert_called_with(env_vars)
    run.assert_called_with('my_framework', env.to_cmd_args(), env_vars)

    write_success_file.assert_called()
    _exit_processes.assert_called_with(0)


@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_with_success(_exit, import_module):
    def success():
        pass

    framework = Mock(entry_point=success)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(_trainer.SUCCESS_CODE)


@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_fails(_exit, import_module):
    def fail():
        raise OSError(os.errno.ENOENT, 'No such file or directory')

    framework = Mock(entry_point=fail)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(errno.ENOENT)


@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_with_client_error(_exit, import_module):
    def fail():
        raise _errors.ClientError(os.errno.ENOENT, 'No such file or directory')

    framework = Mock(entry_point=fail)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(_trainer.DEFAULT_FAILURE_CODE)
