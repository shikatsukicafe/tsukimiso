from torchvision import models
from tsukimiso.aggregator import TkmSrvMixin


class Cifar10Srv(TkmSrvMixin):
    model = models.resnet18()
