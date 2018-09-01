
from protoseg.backends import AbstractBackend


class gluoncv(AbstractBackend):

    def __init__(self):
        AbstractBackend.__init__(self)

    def train(self):
        print('train on gluoncv backend')
