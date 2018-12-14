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

from sagemaker_containers import _mpi, _params, entry_point


def create(env):
    mpi_enabled = env.additional_framework_parameters.get(_params.MPI_ENABLED)
    if mpi_enabled and env.is_master:
        processes_per_host = env.additional_framework_parameters.get(_params.MPI_PROCESSES_PER_HOST,
                                                                     1)
        custom_mpi_options = env.additional_framework_parameters.get(_params.MPI_CUSTOM_OPTIONS, '')

        return _mpi.MasterRunner(env.user_entry_point,
                                 env.to_cmd_args(),
                                 env.to_env_vars(),
                                 env.master_hostname,
                                 env.hosts,
                                 processes_per_host,
                                 custom_mpi_options,
                                 env.network_interface_name)
    elif mpi_enabled:
        return _mpi.WorkerRunner(env.user_entry_point,
                                 env.to_cmd_args(),
                                 env.to_env_vars(),
                                 env.master_hostname)
    else:
        return entry_point.Runner(env.user_entry_point,
                                  env.to_cmd_args(),
                                  env.to_env_vars())
