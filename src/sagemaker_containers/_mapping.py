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

import collections
import itertools

import six

SplitResultSpec = collections.namedtuple('SplitResultSpec', 'included excluded')


def to_env_vars(mapping):  # type: (dict) -> dict
    """Transform a dictionary in a dictionary of env vars.
     Example:
         >>>env_vars = mapping.to_env_vars({'model_dir': '/opt/ml/model', 'batch_size': 25})
         >>>
         >>>print(args)
         ['MODEL_DIR', '/opt/ml/model', 'BATCH_SIZE', 25]
     Args:
         mapping (dict[str, object]): A Python mapping.
     Returns:
         (dict): Dictionary of env vars
     """

    def transform_to_keys(key, value):
        """Transform a key value pair in one or more env keys"""
        keys = {}

        def format_key(name):
            """Decode a key, adds a SAGEMAKER_ prefix to the key and upper case it"""
            if not key:
                return u''

            name = _decode(name).upper()
            return name if name.startswith(u'SAGEMAKER') else u'SAGEMAKER_%s' % name

        # e.f SAGEMAKER_KEY_NAME
        formatted_key = format_key(key)

        # if value is a dictionary, e.g {a:1, b:2}
        if hasattr(value, 'items'):

            # SAGEMAKER_KEY_NAME = a,b
            keys[formatted_key] = u','.join(sorted(value))

            for _k, _v in value.items():
                # SAGEMAKER_KEY_NAME_A
                _name = format_key('%s_%s' % (formatted_key, _k))

                # RECURSIVELY CREATE KEYS
                sub_keys = transform_to_keys(_name, _v)
                keys.update(sub_keys)

        elif isinstance(value, collections.Sequence) and not isinstance(value, six.string_types) and not isinstance(
                value, six.binary_type):

            decoded_seq = sorted([_decode(element) for element in value])

            keys[formatted_key] = u','.join(decoded_seq)
        else:
            keys[formatted_key] = _decode(value)
        return keys

    result = {}

    for k, v in mapping.items():
        result.update(transform_to_keys(k, v))

    return result


def to_cmd_args(mapping):  # type: (dict) -> list
    """Transform a dictionary in a list of cmd arguments.
    Example:
        >>>args = mapping.to_cmd_args({'model_dir': '/opt/ml/model', 'batch_size': 25})
        >>>
        >>>print(args)
        ['--model_dir', '/opt/ml/model', '--batch_size', 25]
    Args:
        mapping (dict[str, object]): A Python mapping.
    Returns:
        (list): List of cmd arguments
    """

    sorted_keys = sorted(mapping.keys())

    def arg_name(obj):
        string = _decode(obj)
        if string:
            return u'--%s' % string if len(string) > 1 else u'-%s' % string
        else:
            return u''

    arg_names = [arg_name(argument) for argument in sorted_keys]

    def arg_value(value):
        if hasattr(value, 'items'):
            map_items = ['%s=%s' % (k, v) for k, v in sorted(value.items())]
            return ','.join(map_items)
        return _decode(value)

    arg_values = [arg_value(mapping[key]) for key in sorted_keys]

    items = zip(arg_names, arg_values)

    return [item for item in itertools.chain.from_iterable(items)]


def _decode(obj):  # type: (bytes or str or unicode or object) -> unicode
    """Decode an object to unicode.
    Args:
        obj (bytes or str or unicode or anything serializable): object to be decoded
    Returns:
        object decoded in unicode.
    """
    if six.PY3 and isinstance(obj, six.binary_type):
        # transforms a byte string (b'') in unicode
        return obj.decode('latin1')
    elif six.PY3:
        # PY3 strings are unicode.
        return str(obj)
    elif isinstance(obj, six.text_type):
        # returns itself if it is unicode
        return obj
    else:
        # decodes pY2 string to unicode
        return str(obj).decode('utf-8')


def split_by_criteria(dictionary, keys=None, prefix=None):  # type: (dict, set or list or tuple) -> SplitResultSpec
    """Split a dictionary in two by the provided keys.

    Args:
        dictionary (dict[str, object]): A Python dictionary
        keys (sequence [str]): A sequence of keys which will be added the split criteria
        prefix (str): A prefix which will be added the split criteria

    Returns:
        `SplitResultSpec` : A collections.namedtuple with the following attributes:

            * Args:
                included (dict[str, object]: A dictionary with the keys included in the criteria.
                excluded (dict[str, object]: A dictionary with the keys not included in the criteria.
    """
    keys = keys or []
    keys = set(keys)

    included_items = {k: dictionary[k] for k in dictionary.keys() if k in keys or (prefix and k.startswith(prefix))}
    excluded_items = {k: dictionary[k] for k in dictionary.keys() if k not in included_items}

    return SplitResultSpec(included=included_items, excluded=excluded_items)


class MappingMixin(collections.Mapping):
    def properties(self):  # type: () -> list
        """
            Returns:
                (list[str]) List of public properties
        """

        _type = type(self)
        return [_property for _property in dir(_type) if self._is_property(_property)]

    def _is_property(self, _property):
        return isinstance(getattr(type(self), _property), property)

    def __getitem__(self, k):
        if not self._is_property(k):
            raise KeyError('Trying to access non property %s' % k)
        return getattr(self, k)

    def __len__(self):
        return len(self.properties())

    def __iter__(self):
        items = {_property: getattr(self, _property) for _property in self.properties()}
        return iter(items)

    def __str__(self):
        return str(dict(self))
