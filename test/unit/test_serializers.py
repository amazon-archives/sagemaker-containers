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
from mock import patch
import numpy as np
from six import b, StringIO

from sagemaker_containers import content_types, serializers

npy = serializers.npy
json = serializers.json
csv = serializers.csv


@patch('numpy.load', lambda x: 'loaded %s' % x)
@patch('sagemaker_containers.serializers.BytesIO', lambda x: 'byte io %s' % x)
def test_npy_loads():
    assert npy.loads(42) == 'loaded byte io 42'


@patch('numpy.save', autospec=True)
@patch('sagemaker_containers.serializers.BytesIO', autospec=True)
def test_npy_dumps(bytes_io, save):
    npy.dumps(42)

    bytes_io.return_value.getvalue.assert_called()
    save.assert_called_with(bytes_io(), 42)


@patch('json.loads', lambda x: 'loaded %s' % x)
def test_json_loads():
    assert json.loads(42) == 'loaded 42'


@patch('json.dumps', lambda x: 'loaded %s' % x)
def test_json_dumps():
    assert json.dumps(42) == 'loaded 42'

    assert json.dumps(np.asarray([42])) == 'loaded [42]'

    assert json.dumps(StringIO('42 is the number')) == 'loaded 42 is the number'


@patch('numpy.genfromtxt', autospec=True)
@patch('sagemaker_containers.serializers.StringIO', autospec=True)
def test_csv_loads(string_io, genfromtxt):
    csv.loads(b('42'))

    string_io.assert_called_with('42')
    genfromtxt.assert_called_with(string_io(), dtype=np.float32, delimiter=',')


@patch('numpy.savetxt', autospec=True)
@patch('sagemaker_containers.serializers.StringIO', autospec=True)
def test_csv_dumps(string_io, savetxt):
    csv.dumps(42)

    string_io.return_value.getvalue.assert_called()
    savetxt.assert_called_with(string_io(), 42, delimiter=',', fmt='%s')


@patch('sagemaker_containers.serializers.npy.dumps', autospec=True)
@patch('sagemaker_containers.serializers.csv.dumps', autospec=True)
@patch('sagemaker_containers.serializers.json.dumps', autospec=True)
def test_dumps(json_dumps, csv_dumps, npy_dumps):
    serializers.dumps(42, content_types.JSON)

    json_dumps.assert_called_once_with(42)

    serializers.dumps(42, content_types.CSV)

    csv_dumps.assert_called_once_with(42)

    serializers.dumps(42, content_types.NPY)

    npy_dumps.assert_called_once_with(42)


@patch('sagemaker_containers.serializers.npy.loads', autospec=True)
@patch('sagemaker_containers.serializers.csv.loads', autospec=True)
@patch('sagemaker_containers.serializers.json.loads', autospec=True)
def test_loads(json_loads, csv_loads, npy_loads):
    serializers.loads(42, content_types.JSON)

    json_loads.assert_called_once_with(42)

    serializers.loads(42, content_types.CSV)

    csv_loads.assert_called_once_with(42)

    serializers.loads(42, content_types.NPY)

    npy_loads.assert_called_once_with(42)
