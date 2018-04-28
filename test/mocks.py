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
import contextlib

import mock


def assert_called_with(mock, **kwargs):
    mock.assert_called_with(**kwargs)
    return mock(**kwargs)


def patch_with_validation(target, *vargs, **kwargs):
    magic_mock = mock.MagicMock(spec=kwargs.get('spec', None))

    def _patch_with_validation(*_vargs, **_kwargs):
        assert vargs == _vargs, 'magic_mock %s invoked with wrong vargs' % magic_mock
        assert kwargs == _kwargs, 'magic_mock %s invoked with wrong kwargs' % magic_mock
        return magic_mock(*_vargs, **_kwargs)

    return mock.patch(target=target, new=_patch_with_validation)


def mock_context_manager(*args, **kwargs):
    @contextlib.contextmanager
    def _mock_context_manager(*v, **k):
        yield mock.MagicMock(*args, **kwargs)(*v, **k)

    return _mock_context_manager


def patch_context_manager(target, *vargs, **kwargs):
    magic_mock = mock.MagicMock(*vargs, **kwargs)

    @contextlib.contextmanager
    def _patch_context_manager(*v, **k):
        yield magic_mock(*v, **k)

    return mock.patch(target=target, new=_patch_context_manager)
