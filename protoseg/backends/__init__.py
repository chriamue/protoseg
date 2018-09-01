# source: http://scottlobdell.me/2015/08/using-decorators-python-automatic-registration/
from __future__ import absolute_import
from os.path import basename, dirname, join
from glob import glob
import importlib.util
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
        b = REGISTERED_BACKENDS[backend]
    except Exception:
        raise Exception(backend + " backend does not exists.")
    return b


def set_backend(backend):
    global _BACKEND
    print('Using backend: ', backend)
    _BACKEND = get_backend(backend)()


def backend():
    return _BACKEND


pwd = dirname(__file__)
for x in glob(join(pwd, '*.py')):
    if not basename(x).startswith('__'):
        spec = importlib.util.spec_from_file_location(basename(x), x)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module_class = getattr(module, basename(x)[:-3])
        register_backend(basename(x)[:-3], module_class)
set_backend('gluoncv')

__all__ = [
    'AbstractBackend',
    'REGISTERED_BACKENDS',
    'backend'
]
