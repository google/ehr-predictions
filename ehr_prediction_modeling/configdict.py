# coding=utf-8
# Copyright 2021 Google Health Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Data structure for storing the configs."""


class ConfigDict(dict):
  """Defines a data structure that stores the configs."""

  def __init__(self, config=None, **kwargs):  # pylint: disable=super-init-not-called
    config = config or dict()
    if kwargs:
      config.update(**kwargs)
    for k, v in config.items():
      setattr(self, k, v)
    for k in self.__class__.__dict__.keys():  # pylint: disable=g-builtin-op
      if (not (k.startswith('__') and k.endswith('__')) and
          not k in ('update', 'pop', 'get')):  # pylint: disable=g-comparison-negation
        setattr(self, k, getattr(self, k))

  def __setattr__(self, key, value):
    if isinstance(value, (list, tuple)):
      value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
    elif isinstance(value, dict) and not isinstance(value, self.__class__):
      value = self.__class__(value)

    if '.' in key:
      raise ValueError('ConfigDict does not accept dots in field names, but '
                       'the key {} contains one.'.format(key))
    else:
      super(ConfigDict, self).__setattr__(key, value)
      super(ConfigDict, self).__setitem__(key, value)

  __setitem__ = __setattr__

  def __getitem__(self, key):
    if '.' in key:
      # As per the check in __setitem__ above, keys cannot contain dots.
      # Hence, we can use dots to do recursive calls.
      key, rest = key.split('.', 1)
      return self[key][rest]
    else:
      return super().__getitem__(key)

  __getattr__ = __getitem__

  def update(self, config=None, **f):
    config = config or dict()
    config.update(f)
    for key, sub_config in config.items():
      if isinstance(sub_config, ConfigDict):
        setattr(self, key, self.get(key, ConfigDict()))
        getattr(self, key).update(sub_config)
      else:
        setattr(self, key, sub_config)

  def get(self, key, default=None):
    """Returns value if key is present, or a user defined value otherwise."""
    try:
      return self[key]
    except KeyError:
      return default
