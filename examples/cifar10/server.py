from tsukimiso.aggregator import TkmSrvMixin
from model import ResNet18

class Cifar10Srv(TkmSrvMixin):
    model = ResNet18()
