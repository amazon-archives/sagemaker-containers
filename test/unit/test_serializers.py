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
import pytest
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


@pytest.mark.parametrize('target, content_type', [
    ('sagemaker_containers.serializers.json.dumps', content_types.JSON),
    ('sagemaker_containers.serializers.csv.dumps', content_types.CSV),
    ('sagemaker_containers.serializers.npy.dumps', content_types.NPY)
])
def test_dumps(target, content_type):
    with patch(target) as serializer:
        serializers.dumps(42, content_type)

        serializer.assert_called_once_with(42)


@pytest.mark.parametrize('target, content_type', [
    ('sagemaker_containers.serializers.json.loads', content_types.JSON),
    ('sagemaker_containers.serializers.csv.loads', content_types.CSV),
    ('sagemaker_containers.serializers.npy.loads', content_types.NPY)
])
def test_dumps(target, content_type):
    with patch(target) as serializer:
        serializers.loads(42, content_type)

        serializer.assert_called_once_with(42)
