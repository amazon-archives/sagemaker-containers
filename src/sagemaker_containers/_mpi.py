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
import argparse
import inspect
import os
import socket
import stat
import subprocess
import time
from typing import Dict, List, Tuple, Any

import libchangehostname
import retrying
from sagemaker_containers import _errors, _logging, _process, _timeout

logger = _logging.get_logger()


def run(cmd,
        env_vars,
        mpi_distribution,
        current_host,
        hosts,
        network_interface_name,
        capture_error):
    # type: (List[str], Dict[str, str], Dict[str, Any], str, List[str], str, bool) -> None
    """Prepares and Executes mpirun in SageMaker in the following order:

    - block instances to wait for MPI execution
    - start the SSHD daemon
    - call mpirun in the master node

        Args:
            cmd (List[str]): Command that will be executed by MPI
            mpi_distribution (Dict[str, str]): Dictionary containing the following MPI
                settings:
                    - custom_mpi_options (List[str]): additional options to be sent to
                      the mpirun call, example: ['--verbose', '--NCCL_DEBUG', 'info'].
                      These options will overwrite any default MPI settings.
                    - processes_per_host (int): number of MPI processes per host (SageMaker
                      instance) that will be created.
            env_vars (Dict[str, str]): A map containing the environment variables to be written.
            current_host (str): Hostname of the current host.
            hosts (str): List with the hostnames of all instances.
            network_interface_name (str): Name of the network interface.
            capture_error (bool): Default false. If True, the running process captures the
                stderr, and appends it to the returned Exception message in case of errors.
        """
    _setup_mpi_environment()

    _create_mpi_script(cmd, _MPI_IS_RUNNING, _MPI_IS_FINISHED)

    if is_master(hosts, current_host):
        logger.info('Starting MPI run as master node')
        _wait_for_worker_nodes_to_start_sshd(hosts)
        _run_mpi_on_all_nodes(hosts,
                              env_vars,
                              mpi_distribution['processes_per_host'],
                              mpi_distribution['custom_mpi_options'],
                              network_interface_name,
                              capture_error=capture_error)
    else:
        logger.info('Starting MPI run as worker node')
        _wait_for_mpi_to_start_running(current_host)
        logger.info('MPI started training process on worker node %s', current_host)
        _wait_until_mpi_stops_running(current_host)


# MPI files.
_MPI_FILES_DIR = '/tmp/sm_mpi'
_MPI_SCRIPT = '/tmp/sm_mpi/mpi_script.sh'
_MPI_IS_RUNNING = '/tmp/sm_mpi/mpi_is_running'
_MPI_IS_FINISHED = '/tmp/sm_mpi/mpi_is_finished'

_SSH_DAEMON_NOT_FOUND_ERROR_MESSAGE = """
SSH daemon not found, please install SSH to allow MPI to communicate different nodes in cluster.

You can install ssh by running following commands:
-------------------------------------------------

1. Install SSH via apt-get:

apt-get update && apt-get install -y --no-install-recommends openssh-server && mkdir -p /var/run/sshd

2. SSH login fix. Otherwise user is kicked off after login:
sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

3. Create SSH key to allow password less ssh between different docker instances:
mkdir -p /root/.ssh/ && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config
"""


def _start_sshd_daemon():  # type: () -> None
    sshd_executable = '/usr/sbin/sshd'

    if not os.path.exists(sshd_executable):
        raise RuntimeError(_SSH_DAEMON_NOT_FOUND_ERROR_MESSAGE)

    subprocess.Popen([sshd_executable, "-D"])


def _setup_mpi_environment():  # type: () -> None
    """Setup MPI environment, i.e. executing change hostname script and starting ssh deamon.
    """
    if not os.path.exists(_MPI_FILES_DIR):
        os.makedirs(_MPI_FILES_DIR)
    _start_sshd_daemon()


def _can_connect(host, port, ssh_socket):  # type: (str,int,socket.socket) -> bool
    """Checks if the connection to provided ``host`` and ``port`` is possible or not.
       Args:
           host (str): Hostname for the host to check connection.
           port (int): Port name of the host to check connection on.
           ssh_socket (socket.socket): SSH Socket to check connection.
    """
    try:
        logger.info('Testing connection to host %s', host)
        ssh_socket.connect((host, port))
        ssh_socket.close()
        logger.info("Can connect to host %s", host)
        return True
    except socket.error:
        logger.info("Can't connect to host %s", host)
        return False


def _create_mpi_script(cmd,
                       mpi_is_running_flag_file,
                       mpi_is_finished_flag_file):
    # type: (list, str, str) -> None
    """Creates a MPI script with user provided information.
        For distributed training: the 'master node' runs mpirun
        with this script, '/mpi_script.sh'. This script creates
        a file '/mpi_is_running' that worker nodes use to determine
        whether training # (started by MPI from the master node) is
        still running. Processes on worker nodes use # /mpi_is_finished
        file to determine when to exit.

    Args:
        args (list): Command line arguments to be passed into customer script.
        train_script (str): Training script to be executed via MPI.
        code_dir (str): Path to directory containing ``train_script``
        mpi_is_running_flag_file (str): Path to the file used to flag the MPI is running status.
        mpi_is_finished_flag_file (str): Path to the file used to flag the MPI is finished status.
    """

    _mpi_script_template = """#!/usr/bin/env bash
    touch %s
    %s
    EXIT_CODE=$?
    touch %s
    exit ${EXIT_CODE}
    """

    content = _mpi_script_template % (mpi_is_running_flag_file, ' '.join(cmd),
                                      mpi_is_finished_flag_file)

    with open(_MPI_SCRIPT, 'w') as w:
        w.write(content)

    st = os.stat(_MPI_SCRIPT)
    os.chmod(_MPI_SCRIPT, st.st_mode | stat.S_IEXEC)

    logger.info('MPI script created at: %s', _MPI_SCRIPT)


def is_master(hosts,
              current_host):  # type: (list, str) -> bool
    """Checks if the current host is master or worker.
    """
    _is_master = current_host == sorted(list(hosts))[0]
    logger.info('Is current host: %s among hosts: %s master: %s', current_host, hosts, _is_master)
    return _is_master


def _wait_for_worker_nodes_to_start_sshd(hosts,
                                         interval=1,
                                         timeout_in_seconds=180):
    # type: (List[str], int, int) -> None
    """Wait for worker nodes to start their ssh daemon to allow MPI communication.
    """
    _hosts = hosts[:]
    with _timeout.timeout(seconds=timeout_in_seconds):
        while _hosts:
            for host in _hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    _hosts.remove(host)
            time.sleep(interval)
        logger.info("Worker node available for communication: {}".format(len(hosts) == 0))


def _run_mpi_on_all_nodes(hosts,
                          env_vars,
                          process_per_host,
                          custom_mpi_options,
                          network_interface_name,
                          capture_error):
    # type: (List[str], Dict[str, str], int, str, str, bool) -> None
    """Run MPI command to execute MPI_SCRIPT on all hosts.
    """
    mpi_command = _mpi_command(hosts,
                               env_vars,
                               process_per_host,
                               custom_mpi_options,
                               network_interface_name)

    _logging.log_script_invocation(mpi_command, env_vars, logger)

    with open(_MPI_SCRIPT) as f:
        logger.info('Running user script:\n\n%s', f.read())

    _process.check_error(mpi_command,
                         _errors.ExecuteUserScriptError,
                         capture_error=capture_error)


def _parse_custom_mpi_options(custom_mpi_options):
    # type: (str) -> Tuple[argparse.Namespace, List[str]]
    """Parse custom MPI options provided by user. Known options default value will be overridden
    and unknown options would be identified separately."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--NCCL_DEBUG', default="INFO", type=str)

    return parser.parse_known_args(custom_mpi_options.split())


def _mpi_command(hosts,
                 env_vars,
                 process_per_host,
                 custom_mpi_options,
                 network_interface_name):
    # type: (List[str], Dict[str, str], int, str, str)-> List[str]
    """Build MPI command with all required MPI flags for sagemaker infrastructure, environment
    variables, provided hyperparameters and custom mpi options.
    """
    num_hosts = len(hosts)
    num_processes = process_per_host * num_hosts

    # By default, use one process per GPU, or one process per node (if training with CPU).
    if process_per_host == 1:
        host_list = hosts
    else:
        host_list = ['%s:%s' % (host, process_per_host) for host in hosts]

    msg = 'Env Hosts: %s Hosts: %s process_per_hosts: %s num_processes: %s'
    logger.info(msg, hosts, host_list, process_per_host, num_processes)

    overridden_known_options, additional_options = _parse_custom_mpi_options(custom_mpi_options)

    logger.info("Network interface name: %s" % network_interface_name)

    command = ['mpirun',
               '--host', ','.join(host_list),
               '-np', str(num_processes),

               '--allow-run-as-root',
               '--display-map',
               '--tag-output',

               '-mca', 'btl_tcp_if_include', network_interface_name,
               '-mca', 'oob_tcp_if_include', network_interface_name,
               '-mca', 'plm_rsh_no_tree_spawn', '1',
               '-mca', 'orte_abort_on_non_zero_status', '1',

               '-x', 'NCCL_SOCKET_IFNAME=%s' % network_interface_name,
               '-x', 'NCCL_DEBUG=%s' % overridden_known_options.NCCL_DEBUG,
               '-x', 'LD_LIBRARY_PATH',
               '-x', 'PATH',
               '-x', 'LD_PRELOAD=%s' % inspect.getfile(libchangehostname),

               ]

    command.extend(additional_options)

    for credential in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']:
        if credential in os.environ:
            command.extend(['-x', credential])

    for name in env_vars:
        command.extend(['-x', name])

    command.append(_MPI_SCRIPT)

    return command


@retrying.retry(stop_max_delay=30000 * 1000,
                wait_fixed=1000,
                retry_on_result=lambda result: result is False)
def _wait_for_mpi_to_start_running(current_host):  # type: (str) -> bool
    """Wait and retry loop until the MPI training starts on this worker.
    """
    logger.debug('Worker node %s is waiting for MPI to start training process', current_host)
    return os.path.isfile(_MPI_IS_RUNNING)


@retrying.retry(wait_fixed=5000,
                retry_on_result=lambda result: result is False)
def _wait_until_mpi_stops_running(current_host):  # type: () -> bool
    logger.debug('Worker node %s is waiting for MPI to finish training process', current_host)
    return os.path.isfile(_MPI_IS_FINISHED)
