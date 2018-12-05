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
import socket
import sys

from mock import call, MagicMock, mock_open, patch
import pkg_resources
import pytest
from six import PY2

import sagemaker_containers
from sagemaker_containers import _errors, _mpi
from sagemaker_containers._mpi import MPIMaster, MPIWorker

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'
_TEST_MPI_SCRIPT_PATH = "/tmp/mpi_script_path"


@patch('os.system')
def test_change_hostname(os_system):
    """Unit tests the ``_change_hostname`` method executes the change-hostname.sh script with valid host or not."""

    host = "any_host"
    _mpi._change_hostname(host)
    os_system.assert_called_with(
        "{} {} {}".format(pkg_resources.resource_filename(sagemaker_containers.__name__, '/bin/change-hostname.sh'),
                          host, _mpi.MPI_FILES_DIR))


@patch("subprocess.Popen")
def test_start_ssh_daemon(subprocess_popen):
    """Unit tests the ``_start_ssh_daemon`` method to verify it is executing the ssh deamon or not"""

    _mpi._start_ssh_daemon()
    subprocess_popen.assert_called_with(["/usr/sbin/sshd", "-D"])


@patch('os.path.exists')
@patch('os.makedirs')
@patch('sagemaker_containers._mpi._change_hostname')
@patch('sagemaker_containers._mpi._start_ssh_daemon')
def test_setup_mpi_environment(start_ssh_daemon, change_hostname, mock_os_mkdirs, mock_os_path_exist):
    """Unit tests the ``_setup_mpi_environment`` method to verify all steps are performed for mpi setup"""

    mock_env = mock_training_env()
    mock_os_path_exist.return_value = False

    _mpi._setup_mpi_environment(mock_env.current_host)

    mock_os_path_exist.assert_called_with(_mpi.MPI_FILES_DIR)
    mock_os_mkdirs.assert_called_with(_mpi.MPI_FILES_DIR)
    change_hostname.assert_called_with(current_host=mock_env.current_host)
    start_ssh_daemon.assert_called()


def test_can_connect():
    mock_socket = MagicMock(spec=['connect', 'close'])
    mock_socket.connect.side_effect = [socket.error('expected'), socket.error('expected'), None]

    first_call = _mpi._can_connect('algo-2', 2222, mock_socket)
    second_call = _mpi._can_connect('algo-2', 2222, mock_socket)
    third_call = _mpi._can_connect('algo-2', 2222, mock_socket)

    assert not first_call
    assert not second_call
    assert third_call
    assert mock_socket.connect.call_count == 3


def test_create_mpi_script():
    """Unit test for ``_create_mpi_script``, to verify the script is generated in the valid format.
    """

    mpi_script_path = "/tmp/mpi_script.sh"
    _mpi._create_mpi_script(args=["--sample", "arg1", "--sample2", "1"], train_script="train.py",
                            code_dir="/opt/ml/code", mpi_script_path=mpi_script_path,
                            mpi_is_running_flag_file="/tmp/mpi_is_running",
                            mpi_is_finished_flag_file="/tmp/mpi_is_finished")

    with open(mpi_script_path, 'r') as mpi_script_file:
        content = mpi_script_file.read()
        assert """#!/usr/bin/env bash
touch /tmp/mpi_is_running
%s /opt/ml/code/train.py --sample arg1 --sample2 1
EXIT_CODE=$?
touch /tmp/mpi_is_finished
exit ${EXIT_CODE}
""" % (sys.executable) == content


def mock_training_env(current_host='algo-1', hosts=[], hyperparameters=None,
                      module_dir='s3://my/script', module_name='imagenet', **kwargs):
    hosts = hosts or ['algo-1']

    hyperparameters = hyperparameters or {}

    return MagicMock(current_host=current_host, hosts=hosts, hyperparameters=hyperparameters,
                     module_dir=module_dir, module_name=module_name, network_interface_name="ethwe", **kwargs)


# MPI Master Tests

@patch('sagemaker_containers._mpi._can_connect', side_effect=[False, False, True])
@patch('time.sleep')
@patch('socket.socket')
def test_wait_for_worker_nodes_to_start_sshd(socket, sleep, _can_connect):
    mpi_master = MPIMaster(env=mock_training_env(),
                           process_per_host=1,
                           mpi_script_path=_TEST_MPI_SCRIPT_PATH,
                           custom_mpi_options="")
    mpi_master._wait_for_worker_nodes_to_start_sshd(hosts=['algo-2'])

    assert _can_connect.call_count == 3
    socket.assert_called()
    sleep.assert_called()


def test_parse_custom_mpi_options():
    mpi_master = MPIMaster(env=mock_training_env(),
                           process_per_host=1,
                           custom_mpi_options="--NCCL_DEBUG WARN --Dummy dummyvalue",
                           mpi_script_path=_TEST_MPI_SCRIPT_PATH)

    known_args, unknown_args = mpi_master._parse_custom_mpi_options()

    assert known_args.NCCL_DEBUG == "WARN"
    assert unknown_args == ["--Dummy", "dummyvalue"]


@patch(builtins_open, mock_open())
@patch('sagemaker_containers._mpi.MPIMaster._build_mpi_command')
@patch('sagemaker_containers._process.check_error')
@patch('sagemaker_containers._process.create')
@patch('sagemaker_containers._logging.log_script_invocation')
@pytest.mark.parametrize('wait, capture_error',
                         [(True, False),
                          (True, True),
                          (False, False),
                          (False, True)])
def test_run_mpi_on_all_nodes(_log_script_invocation, _process_create, _process_check_error, _build_mpi_command, wait,
                              capture_error):
    cmd = "mpirun -np 2"
    _build_mpi_command.return_value = cmd

    mock_env = mock_training_env()
    mpi_master = MPIMaster(env=mock_env,
                           process_per_host=1,
                           mpi_script_path=_TEST_MPI_SCRIPT_PATH,
                           custom_mpi_options="")
    mpi_master._run_mpi_on_all_nodes(wait, capture_error)

    _build_mpi_command.assert_called()
    _log_script_invocation.assert_called()
    open.assert_any_call(_TEST_MPI_SCRIPT_PATH)
    if wait:
        _process_check_error.assert_called_with(cmd.split(), _errors.ExecuteUserScriptError,
                                                capture_error=capture_error)
    else:
        _process_create.assert_called_with(cmd.split(), _errors.ExecuteUserScriptError, capture_error=capture_error)


def test_is_master():
    assert _mpi.is_master(["algo-1", "algo-2"], "algo-1")
    assert not _mpi.is_master(["algo-1", "algo-2"], "algo-2")


@patch('sagemaker_containers._mpi.MPIMaster._wait_for_worker_nodes_to_start_sshd')
@patch('sagemaker_containers._mpi.MPIMaster._run_mpi_on_all_nodes')
@pytest.mark.parametrize('wait, capture_error',
                         [(True, False),
                          (True, True),
                          (False, False),
                          (False, True)])
def test_mpi_master_call(_run_mpi_on_all_nodes, _wait_for_worker_nodes_to_start_sshd, wait, capture_error):
    mock_env = mock_training_env()
    mpi_master = MPIMaster(env=mock_env,
                           process_per_host=1,
                           mpi_script_path=_TEST_MPI_SCRIPT_PATH,
                           custom_mpi_options="")
    mpi_master.run(wait, capture_error)

    _run_mpi_on_all_nodes.assert_called_with(wait, capture_error)
    assert _wait_for_worker_nodes_to_start_sshd.call_count == 1


# MPI Worker Tests

def test_mpi_worker_init():
    current_host = "algo-1"
    mpi_worker = MPIWorker(current_host=current_host)
    assert mpi_worker._current_host == current_host


def test_wait_for_mpi_to_start_running():
    with patch('os.path.isfile') as mock_isfile, patch('time.sleep'):
        mock_isfile.side_effect = [False, False, True]

        mpi_worker = MPIWorker(current_host="algo-1")
        mpi_worker._wait_for_mpi_to_start_running()
        mock_isfile.assert_has_calls(
            [call(_mpi._MPI_IS_RUNNING), call(_mpi._MPI_IS_RUNNING),
             call(_mpi._MPI_IS_RUNNING)])

        assert len(mock_isfile.call_args_list) == 3


def test_wait_until_mpi_stops_running():
    with patch('os.path.isfile') as mock_isfile, patch('time.sleep'):
        mock_isfile.side_effect = [False, False, True]

        mpi_worker = MPIWorker(current_host="algo-1")
        mpi_worker._wait_until_mpi_stops_running()

        mock_isfile.assert_has_calls(
            [call(_mpi._MPI_IS_FINISHED), call(_mpi._MPI_IS_FINISHED),
             call(_mpi._MPI_IS_FINISHED)])
        assert mock_isfile.call_count == 3


@patch('sagemaker_containers.training_env')
@patch('sagemaker_containers._mpi._setup_mpi_environment')
@patch('sagemaker_containers._mpi._create_mpi_script', autospec=True)
@patch('sagemaker_containers._mpi.MPIMaster')
@pytest.mark.parametrize('train_script, code_dir, args, env_vars, wait, capture_error',
                         [("train.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, True, False),
                          ("train1.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, True, True),
                          ("train.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, False, False),
                          ("train.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, False, True)])
def test_mpi_run_for_master(mock_master, _create_mpi_script, _setup_mpi_environment, mock_env_generator, train_script,
                            code_dir, args, env_vars, wait, capture_error):
    mock_num_of_processes_per_host = MagicMock()
    mock_custom_mpi_options = MagicMock()
    mock_env = mock_training_env()
    mock_env.additional_framework_parameters.get.side_effect = [mock_num_of_processes_per_host, mock_custom_mpi_options]
    mock_master_instance = mock_master.return_value
    mock_master_instance.is_master.side_effect = [True]

    mock_env_generator.return_value = mock_env

    _mpi.mpi_run(train_script, code_dir, args, env_vars, wait, capture_error)

    assert _setup_mpi_environment.call_count == 1
    _create_mpi_script.assert_called_with(args, train_script, code_dir, _mpi._MPI_SCRIPT,
                                          _mpi._MPI_IS_RUNNING, _mpi._MPI_IS_FINISHED)

    mock_master.assert_called_with(mock_env, mock_num_of_processes_per_host, _mpi._MPI_SCRIPT, mock_custom_mpi_options)
    mock_master_instance.run.assert_called_with(wait, capture_error)


@patch('sagemaker_containers.training_env')
@patch('sagemaker_containers._mpi._setup_mpi_environment')
@patch('sagemaker_containers._mpi._create_mpi_script')
@patch('sagemaker_containers._mpi.MPIMaster', autospec=True)
@patch('sagemaker_containers._mpi.MPIWorker', autospec=True)
@patch('sagemaker_containers._mpi.is_master')
@pytest.mark.parametrize('train_script, code_dir, args, env_vars, wait, capture_error',
                         [("train.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, True, False),
                          ("train1.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, True, True),
                          ("train.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, False, False),
                          ("train.py", "/opt/ml/code", ["--sample", "arg1"], {"SM_SAMPLE": "VAL1"}, False, True)])
def test_mpi_run_for_worker(mock_is_master, mock_worker, mock_master, _create_mpi_script, _setup_mpi_environment,
                            mock_env_generator,
                            train_script, code_dir, args, env_vars, wait, capture_error):
    mock_num_of_processes_per_host = MagicMock()
    mock_custom_mpi_options = MagicMock()
    mock_env = mock_training_env()
    mock_env.additional_framework_parameters.get.side_effect = [mock_num_of_processes_per_host, mock_custom_mpi_options]

    mock_master_instance = mock_master.return_value
    mock_is_master.side_effect = [False]

    mock_worker_instance = mock_worker.return_value

    mock_env_generator.return_value = mock_env

    _mpi.mpi_run(train_script, code_dir, args, env_vars, wait, capture_error)

    assert _setup_mpi_environment.call_count == 1
    _create_mpi_script.assert_called_with(args, train_script, code_dir, _mpi._MPI_SCRIPT,
                                          _mpi._MPI_IS_RUNNING, _mpi._MPI_IS_FINISHED)

    mock_master.assert_called_with(mock_env, mock_num_of_processes_per_host, _mpi._MPI_SCRIPT, mock_custom_mpi_options)
    mock_master_instance.run.assert_not_called()

    mock_worker.assert_called_with(mock_env)
    mock_worker_instance.run.assert_called()
