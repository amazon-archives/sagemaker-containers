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
import os
import shlex
import socket
import stat
import subprocess
import sys
import time

import pkg_resources
from retrying import retry

import sagemaker_containers
from sagemaker_containers import _errors, _logging, _params, _process, _timeout

logger = _logging.get_logger()

_CHANGE_HOSTNAME_FILE_PATH = pkg_resources.resource_filename(sagemaker_containers.__name__, '/bin/change-hostname.sh')

_SSHD_EXECUTABLE_PATH = '/usr/sbin/sshd'

_SKIP_ENV_VARS_TO_COPY_IN_MPIRUN = ["SM_CURRENT_HOST"]

# MPI files.
MPI_FILES_DIR = "/tmp/sm_mpi"
_MPI_SCRIPT = "/tmp/sm_mpi/mpi_script.sh"
_MPI_IS_RUNNING = "/tmp/sm_mpi/mpi_is_running"
_MPI_IS_FINISHED = "/tmp/sm_mpi/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/tmp/sm_mpi/libchangehostname.so"

_MPI_COMMAND_TEMPLATE = "mpirun --host {}" \
                        + " -np {} " \
                        + " --allow-run-as-root" \
                        + " --display-map" \
                        + " --tag-output" \
                        + " -mca btl_tcp_if_include {}" \
                        + " -mca oob_tcp_if_include {}" \
                        + " -x NCCL_SOCKET_IFNAME={}" \
                        + " --mca plm_rsh_no_tree_spawn 1" \
                        + " -mca orte_abort_on_non_zero_status 1" \
                        + " -x NCCL_DEBUG={}" \
                        + " -x LD_LIBRARY_PATH -x PATH" \
                        + " -x LD_PRELOAD={}" \
                        + " {}"

_MPI_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
touch %s
%s
EXIT_CODE=$?
touch %s
exit ${EXIT_CODE}
"""

_SSH_DEAMON_NOT_FOUND_ERROR_MESSAGE = """
SSH deamon not found, please install SSH to allow MPI to communicate different nodes in cluster.

You can install ssh by running following commands:
-------------------------------------------------

1. Install SSH via apt-get:

apt-get update && apt-get install -y --no-install-recommends openssh-server && mkdir -p /var/run/sshd

2. SSH login fix. Otherwise user is kicked off after login:
sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

3. Create SSH key to allow password less ssh between diffferent docker instances:
mkdir -p /root/.ssh/ && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config
"""


def _change_hostname(current_host):  # type: (str) -> None
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.
    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("{} {} {}".format(_CHANGE_HOSTNAME_FILE_PATH, current_host, MPI_FILES_DIR))


def _start_ssh_daemon():  # type: () -> None
    """Starts the ssh deamon
    """
    exists = os.path.isfile(_SSHD_EXECUTABLE_PATH)
    if not exists:
        raise RuntimeError(_SSH_DEAMON_NOT_FOUND_ERROR_MESSAGE)

    subprocess.Popen([_SSHD_EXECUTABLE_PATH, "-D"])


def _setup_mpi_environment(current_host):  # type: (str) -> None
    """Setup MPI environment, i.e. executing change hostname script and starting ssh deamon.
       Args:
           current_host (str): Current host name.
    """
    if not os.path.exists(MPI_FILES_DIR):
        os.makedirs(MPI_FILES_DIR)
    _change_hostname(current_host=current_host)
    _start_ssh_daemon()


def _can_connect(host, port, ssh_socket):  # type: (str,int,socket.socket) -> bool
    """Checks if the connection to provided ``host`` and ``port`` is possible or not.
       Args:
           host (str): Hostname for the host to check connection.
           port (int): Port name of the host to check connection on.
           ssh_socket (socket.socket): SSH Socket to check connection.
    """
    try:
        logger.info("Testing connection to host {}".format(host))
        ssh_socket.connect((host, port))
        ssh_socket.close()
        logger.info("Can connect to host {}".format(host))
        return True
    except socket.error:
        logger.info("Can't connect to host {}".format(host))
        return False


def _create_mpi_script(args, train_script, code_dir, mpi_script_path, mpi_is_running_flag_file,
                       mpi_is_finished_flag_file):  # type: (list, str, str, str, str, str) -> None
    """Creates a MPI script with user provided information.
        For distributed training: the 'master node' runs mpirun with this script, '/mpi_script.sh'.
        This script creates a file '/mpi_is_running' that worker nodes use to determine whether training # (started by
        MPI from the master node) is still running. Processes on worker nodes use # /mpi_is_finished file to determine
        when to exit.
    Args:
        args (list): Command line arguments to be passed into customer script.
        train_script (str): Training script to be executed via MPI.
        code_dir (str): Path to directory containing ``train_script``
        mpi_script_path (str): Path where the MPI script is created.
        mpi_is_running_flag_file (str): Path to the file used to flag the MPI is running status.
        mpi_is_finished_flag_file (str): Path to the file used to flag the MPI is finished status.
    """

    python_cmd = [sys.executable, "{}/{}".format(code_dir, train_script)]
    python_cmd.extend(args)

    content = _MPI_SCRIPT_TEMPLATE % (mpi_is_running_flag_file,
                                      ' '.join(python_cmd),
                                      mpi_is_finished_flag_file)

    with open(mpi_script_path, 'w') as w:
        w.write(content)

    st = os.stat(mpi_script_path)
    os.chmod(mpi_script_path, st.st_mode | stat.S_IEXEC)

    logger.info("MPI script created at: {}".format(mpi_script_path))


def is_master(hosts, current_host):  # type: (list, str) -> bool
    """Checks if the current host is master or worker.
    """
    _is_master = current_host == sorted(list(hosts))[0]
    logger.info("Is current host: {} among hosts: {} master: {}".format(current_host, hosts, _is_master))
    return _is_master


class MPIMaster(object):
    """MPI Master, defines all the operations that are executed at the master node to execute MPI job. It coordinates
        with worker nodes to see if they are ready for MPI, then builds the MPI command and execute it to launch job on
        all workers.
        Args:
            env (TrainingEnv): an instance of the training environment.
            process_per_host (int): Number of processes per host to be executed by MPI
            mpi_script_path (str): Path where the MPI script is created.
            custom_mpi_options (str): Custom MPI options provided by user, this string will be parsed into arguments and
             all known argument's value will override the defaults whereas unknown arguments will be appended directly
             to MPI command.
    """

    def __init__(self, env, process_per_host, mpi_script_path, custom_mpi_options):
        self.env = env
        self.process_per_host = process_per_host
        self.mpi_script_path = mpi_script_path
        self.custom_mpi_options = custom_mpi_options

    def _wait_for_worker_nodes_to_start_sshd(self, hosts, interval=1,
                                             timeout_in_seconds=180):  # type: (list, int, int) -> None
        """Wait for worker nodes to start their ssh deamon to allow MPI communication.
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

    def _run_mpi_on_all_nodes(self, wait, capture_error):  # type: (bool,bool) -> None
        """Run MPI command to execute MPI_SCRIPT on all hosts.
        """
        mpi_command = self._build_mpi_command()
        cmd = shlex.split(mpi_command)

        _logging.log_script_invocation(cmd, self.env.to_env_vars(), logger)

        with open(self.mpi_script_path) as f:
            logger.info('Running user script:\n\n%s', f.read())

        logger.info("Executing mpi command with wait: {} capture_error: {}".format(wait, capture_error))

        if wait:
            return _process.check_error(cmd, _errors.ExecuteUserScriptError, capture_error=capture_error)

        else:
            return _process.create(cmd, _errors.ExecuteUserScriptError, capture_error=capture_error)

    def _parse_custom_mpi_options(self):  # type: () -> (ArgumentParser,list)
        """Parse custom MPI options provided by user. Known options default value will be overriden and unknown options
        would be identified separately."""

        parser = argparse.ArgumentParser()
        parser.add_argument('--NCCL_DEBUG', default="INFO", type=str)

        return parser.parse_known_args(self.custom_mpi_options.split())

    def _build_mpi_command(self):  # type: ()-> None
        """Build MPI command with all required MPI flags for sagemaker infrastructure, environment variables, provided
        hyperparameters and custom mpi options.
        """
        num_hosts = len(self.env.hosts)
        num_processes = self.process_per_host * num_hosts

        # By default, use one process per GPU, or one process per node (if training with CPU).
        if self.process_per_host == 1:
            host_list = self.env.hosts
        else:
            host_list = [host + ':{}'.format(self.process_per_host) for host in self.env.hosts]

        logger.info("Env Hosts: {} Hosts: {} process_per_hosts: {} num_processes: {}".format(self.env.hosts, host_list,
                                                                                             self.process_per_host,
                                                                                             num_processes))

        overriden_known_options, additional_options = self._parse_custom_mpi_options()

        interface_name = self.env.network_interface_name

        logger.info("Network interface name: {}".format(interface_name))

        mpi_command = _MPI_COMMAND_TEMPLATE.format(",".join(host_list), num_processes, interface_name, interface_name,
                                                   interface_name, overriden_known_options.NCCL_DEBUG,
                                                   _CHANGE_HOSTNAME_LIBRARY, " ".join(additional_options))

        credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
        for v in credential_vars:
            if v in os.environ:
                mpi_command += " -x {}".format(v)

        for name, value in self.env.to_env_vars().items():
            if name not in _SKIP_ENV_VARS_TO_COPY_IN_MPIRUN:
                mpi_command += ' -x {}="{}"'.format(name, value)

        mpi_command += " {}".format(self.mpi_script_path)

        return mpi_command

    def run(self, wait, capture_error):  # type: (bool, bool) -> None
        """Executes the master's node operation
            Args:
                wait (bool): If True, holds the process executing the user entry-point.
                             If False, returns the process that is executing it.
                capture_error (bool): Default false. If True, the running process captures the
                    stderr, and appends it to the returned Exception message in case of errors.
        """
        self._wait_for_worker_nodes_to_start_sshd(self.env.hosts)
        self._run_mpi_on_all_nodes(wait, capture_error)


class MPIWorker(object):
    """ MPI Worker, defines all operations that are executed at the worker node. It waits for the MPI job to start and
        then it waits for the MPI job to finish.
        Args:
            current_host (str): Current host id.
    """

    def __init__(self, current_host):
        self._current_host = current_host

    @retry(stop_max_delay=30000 * 1000, wait_fixed=1000, retry_on_result=lambda result: result is False)
    def _wait_for_mpi_to_start_running(self):  # type: () -> None
        """Wait and retry loop until the MPI training starts on this worker.
        """
        return os.path.isfile(_MPI_IS_RUNNING)

    @retry(wait_fixed=5000, retry_on_result=lambda result: result is False)
    def _wait_until_mpi_stops_running(self):  # type: () -> None
        """Wait and retry loop until the MPI training is finished on this worker.
        """
        return os.path.isfile(_MPI_IS_FINISHED)

    def run(self):  # type: () -> None
        logger.info("Worker node {} is waiting for MPI to start training process".format(self._current_host))
        self._wait_for_mpi_to_start_running()

        logger.info("MPI started training process on worker node {}".format(self._current_host))

        self._wait_until_mpi_stops_running()
        logger.info("Training process started by MPI on worker node {} stopped".format(self._current_host))


def mpi_run(train_script, code_dir, args, env_vars, wait,
            capture_error):  # type: (str, str, list, dict, bool, bool) -> None
    """It runs the mpi command to launch user provided training script.
        Args:
            train_script (str): Train script to executed by the ``MPILauncher``
            code_dir (str): Path to directory containing ``train_script``
            args (list):  A list of program arguments.
            env_vars (dict): A map containing the environment variables to be written.
            wait (bool): If True, holds the process executing the user entry-point.
                         If False, returns the process that is executing it.
            capture_error (bool): Default false. If True, the running process captures the
                stderr, and appends it to the returned Exception message in case of errors.
        """
    env = sagemaker_containers.training_env()

    num_of_processes_per_host = env.additional_framework_parameters.get(_params.SAGEMAKER_MPI_NUM_PROCESSES_PER_HOST, 1)
    custom_mpi_options = env.additional_framework_parameters.get(_params.SAGEMAKER_MPI_CUSTOM_MPI_OPTIONS, "")

    logger.info("MPI requested for train_script: {} code_dir: {} process per hosts: {} and custom_mpi_options: {}"
                .format(train_script, code_dir, num_of_processes_per_host, custom_mpi_options))

    _setup_mpi_environment(env.current_host)

    _create_mpi_script(args, train_script, code_dir, _MPI_SCRIPT, _MPI_IS_RUNNING, _MPI_IS_FINISHED)

    mpi_master = MPIMaster(env, num_of_processes_per_host, _MPI_SCRIPT, custom_mpi_options)
    if is_master(env.hosts, env.current_host):
        logger.info("Inside Master")
        mpi_master.run(wait, capture_error)
    else:
        logger.info("Inside Worker")
        MPIWorker(env).run()
