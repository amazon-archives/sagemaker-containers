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
# import json
# import os
import sys

import pytest

from sagemaker_containers import _errors, _process

#
# def test_libchangehostname_with_env_set():
#     # if os.path.exists("/opt/ml/input/config/"):
#     #     os.removedirs("/opt/ml/input/config/")
#
#     # os.makedirs("/opt/ml/input/config")
#
#     with open("/opt/ml/input/config/resourceconfig.json", 'w') as f:
#         json.dump({'current_host': 'algo-5'}, f)
#
#     py_cmd = "import libchangehostname\nassert libchangehostname.call(30) == 'algo-5'"
#     _process.check_error([sys.executable, '-c', py_cmd], _errors.ExecuteUserScriptError)


def test_libchangehostname_with_env_not_set():
    py_cmd = "import libchangehostname\nassert libchangehostname.call(30) == 'algo-9'"

    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.check_error([sys.executable, '-c', py_cmd], _errors.ExecuteUserScriptError)
