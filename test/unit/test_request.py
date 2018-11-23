import numpy as np
from mock import patch
import pytest

from sagemaker_containers import _content_types, _encoders, _worker
import test


@pytest.mark.parametrize('content_type_header', ['ContentType', 'Content-Type'])
def test_request(content_type_header):
    headers = {
        content_type_header: _content_types.JSON,
        'Accept': _content_types.CSV
    }

    request = _worker.Request(test.environ(data='42', headers=headers))

    assert request.content_type == _content_types.JSON
    assert request.accept == _content_types.CSV
    assert request.content == '42'

    headers = {content_type_header: _content_types.NPY, 'Accept': _content_types.CSV}
    request = _worker.Request(test.environ(data=_encoders.encode([6, 9.3], _content_types.NPY),
                                           headers=headers))

    assert request.content_type == _content_types.NPY
    assert request.accept == _content_types.CSV

    result = _encoders.decode(request.data, _content_types.NPY)
    np.testing.assert_array_equal(result, np.array([6, 9.3]))


@patch('sagemaker_containers._env.ServingEnv')
def test_request_without_accept(serving_env):
    serving_env.default_accept = 'application/json'

    request = _worker.Request(test.environ(), serving_env=serving_env)
    assert request.accept == 'application/json'


@patch('sagemaker_containers._env.ServingEnv')
def test_request_with_accept_any(serving_env):
    serving_env.default_accept = 'application/NPY'

    request = _worker.Request(test.environ(headers={'Accept': _content_types.ANY}),
                              serving_env=serving_env)

    assert request.accept == 'application/NPY'


@patch('sagemaker_containers._env.ServingEnv')
def test_request_with_accept(serving_env):
    serving_env.default_accept = 'application/NPY'

    request = _worker.Request(test.environ(headers={'Accept': _content_types.CSV}),
                              serving_env=serving_env)

    assert request.accept == 'text/csv'


@pytest.mark.parametrize('content_type_header', ['ContentType', 'Content-Type'])
def test_request_content_type(content_type_header):
    response = test.request(headers={content_type_header: _content_types.CSV})
    assert response.content_type == _content_types.CSV

    response = _worker.Request(test.environ(headers={'ContentType': _content_types.NPY}))
    assert response.content_type == _content_types.NPY
