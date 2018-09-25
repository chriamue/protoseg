# source: http://scottlobdell.me/2015/08/using-decorators-python-automatic-registration/
from __future__ import absolute_import
from os.path import basename, dirname, join
from glob import glob
import importlib.util
from importlib import import_module
import sys

from .__abstract_backend import AbstractBackend

REGISTERED_BACKENDS = {}
_BACKEND = None


def register_backend(name, backend):
    REGISTERED_BACKENDS[name] = backend
    print('registered backend:', name)
    return backend


def get_backend(backend):
    try:
        full_class = REGISTERED_BACKENDS[backend]
        module_name, class_name = full_class.rsplit('.', 1)
        mod = import_module(module_name)
        clss = getattr(mod, class_name)()
    except Exception:
        raise Exception(backend + " backend does not exists.")
    return clss


def set_backend(backend):
    global _BACKEND
    print('Using backend: ', backend)
    _BACKEND = get_backend(backend)


def backend():
    return _BACKEND


pwd = dirname(__file__)
for x in glob(join(pwd, '*.py')):
    if not basename(x).startswith('__'):
        class_name = basename(x)[:-3]
        full_class = __package__ + '.' + class_name + '.' + class_name
        register_backend(class_name, full_class)
set_backend('gluoncv_backend')

__all__ = [
    'AbstractBackend',
    'REGISTERED_BACKENDS',
    'backend'
]
