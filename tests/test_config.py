import pytest
from protoseg import Config

configs = {'run1':{},
            'run2':{}}

def test_len():
    config = Config(configs=configs)
    assert(len(config), 2)

def test_index():
    config = Config(configs=configs)
    assert(config[0], 'run1')
    assert(config[1], 'run2')

def test_iterator():
    count = 0
    config = Config(configs=configs)
    for _ in config:
        count += 1
    assert(count, 2)