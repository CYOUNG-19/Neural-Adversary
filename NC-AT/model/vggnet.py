import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models, datasets, transforms

class VGG13_32x32(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG13_32x32, self).__init__()
        self.features = models.vgg13(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 4096)),
            ('relu1', nn.ReLU(True)),
            ('dropout1', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU(True)),
            ('dropout2', nn.Dropout()),
            ('fc3', nn.Linear(4096, num_classes))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16_64x64(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_64x64, self).__init__()
        self.features = models.vgg16(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 64x64 输入经过 VGG-16 后特征图大小为 2x2
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512 * 2 * 2, 4096)),
            ('relu1', nn.ReLU(True)),
            ('dropout1', nn.Dropout()),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU(True)),
            ('dropout2', nn.Dropout()),
            ('fc3', nn.Linear(4096, num_classes))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def return_vgg(num_classes, dataset):
    if dataset == 'imagenet':
        return VGG16_64x64(num_classes)
    else:
        return VGG13_32x32(num_classes)

