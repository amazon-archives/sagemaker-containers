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
import importlib
import os
import traceback

from sagemaker_containers import env


def report_training_status(train_func):

    def train_and_report(*args, **kwargs):
        training_env = env.TrainingEnv()
        exit_code = 0

        try:
            train_func(*args, **kwargs)
            training_env.write_success_file()
        except Exception as e:
            exit_code = 1 if not hasattr(e, 'errno') else e.errno
            failure_msg = 'Exception caught in training: {}\n{}\n'.format(e, traceback.format_exc())
            training_env.write_failure_file(failure_msg)
            raise e
        finally:
            os._exit(exit_code)

    return train_and_report


def train():
    training_env = env.TrainingEnv()

    # TODO: iquintero - add error handling for ImportError to let the user know
    # if the framework module is not defined.
    framework_name, entry_point = training_env.framework_module.split(':')
    framework = importlib.import_module(framework_name)
    entry = getattr(framework, entry_point)
    entry()
