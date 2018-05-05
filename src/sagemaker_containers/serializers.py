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

import json as stdjson
import textwrap

import numpy as np
from six import BytesIO, StringIO

from sagemaker_containers import content_types


class Npy(object):
    """Npy Serializer"""

    @staticmethod
    def loads(obj):  # type: (object) -> object
        """Load arrays or pickled objects from memory into numpy arrays.

        Args:
            obj (object): object to be deserialized.

        Returns:
            (np.array): deserialized numpy array.
        """
        stream = BytesIO(obj)
        return np.load(stream)

    @staticmethod
    def dumps(nparray):  # type: (np.array) -> object
        """Serializes an numpy array into memory.

        Args:
            nparray (np.array): np array object to be serialized.

        Returns:
            (obj): serialized np array.
        """
        buffer = BytesIO()
        np.save(buffer, nparray)
        return buffer.getvalue()


class Json(object):
    """JSON serializer"""

    @staticmethod
    def loads(string):  # type: (str) -> object
        """Deserialize a JSON string in a Python object.

        Args:
            string (str): JSON string to be deserialized

        Returns:
            (object): deserialized Python object.
        """
        return stdjson.loads(string)

    @staticmethod
    def _numpy_dumps(obj):
        try:
            return stdjson.dumps(obj.tolist())
        except (TypeError, SyntaxError, AttributeError):
            return None

    @staticmethod
    def _stream_dumps(obj):
        try:
            return stdjson.dumps(obj.read())
        except (TypeError, SyntaxError, AttributeError):
            return None

    def dumps(self, obj):  # type: (object) -> str
        """Serializes python objects, streams, numpy arrays into JSON.

        Args:
            obj (object): object to be serialized.

        Returns:
            str:  serialized object
        """
        return self._numpy_dumps(obj) or self._stream_dumps(obj) or stdjson.dumps(obj)


class Csv(object):
    """CSV serializer"""

    @staticmethod
    def loads(string):  # type: (str) -> object
        """Deserialize a CSV string in a Python object.

        Args:
            string (str): CSV string to be deserialized

        Returns:
            (object): deserialized Python object.
        """
        try:
            string = string.decode('utf-8')
        except AttributeError:
            pass
        stream = StringIO(string)
        return np.genfromtxt(stream, dtype=np.float32, delimiter=',')

    @staticmethod
    def dumps(obj):  # type: (object) -> str
        """Serializes python objects, streams, numpy arrays into CSV.

        Args:
            obj (object): object to be serialized.

        Returns:
            str:  deserialized object.
        """
        stream = StringIO()
        np.savetxt(stream, obj, delimiter=',', fmt='%s')
        return stream.getvalue()


def dumps(obj, content_type):
    """Serializes an object to a one of the default content types.

    Args:
        obj (object): to be serialized.
        content_type (str): content type to be used.

    Returns:
        object: serialized object.
    """
    return _SerializerMap(content_type).dumps(obj)


def loads(obj, content_type):
    """Deserialize an object to a one of the default content types.

    Args:
        obj (object): to be deserialized.
        content_type (str): content type to be used.

    Returns:
        object: deserialized object.
    """
    return _SerializerMap(content_type).loads(obj)


npy = Npy()
csv = Csv()
json = Json()


class _SerializerMap(object):
    map = {content_types.JSON: json,
           content_types.CSV: csv,
           content_types.NPY: npy}

    def __init__(self, content_type):
        try:
            self._serializer = self.map[content_type]
        except KeyError:
            raise UnsupportedFormatError(self._content_type)

    def __getattr__(self, name):
        return getattr(self._serializer, name, None)


class UnsupportedFormatError(Exception):
    def __init__(self, content_type, **kwargs):
        super(Exception, self).__init__(**kwargs)
        self.message = textwrap.dedent(
            """Content type %s is not supported by this framework.

               Please implement input_fn to to deserialize the request data or an output_fn to serialize the
               response. For more information: https://github.com/aws/sagemaker-python-sdk#input-processing"""
            % content_type)
