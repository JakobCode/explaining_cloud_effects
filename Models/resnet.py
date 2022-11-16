"""
This script is taken from the official SEN12MS repository:
https://github.com/schmitt-muc/SEN12MS/blob/master/classification/models/ResNet.py
"""


# Modified from Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Modified by Yu-Lun Wu, TUM
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data)


def fc_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)


class ResNet18(nn.Module):
    """ResNet18 implementation"""
    def __init__(self, n_inputs=12, num_classes=17):
        super().__init__()

        resnet = models.resnet18(pretrained=False)

        self.conv1 = nn.Conv2d(
            n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(512, num_classes)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        return logits


class ResNet34(nn.Module):
    """ResNet34 implementation"""
    def __init__(self, n_inputs=12, num_classes=17):
        super().__init__()

        resnet = models.resnet34(pretrained=False)

        self.conv1 = nn.Conv2d(
            n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(512, num_classes)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        return logits


class ResNet50(nn.Module):
    """ResNet50 implementation"""
    def __init__(self, n_inputs=12, num_classes=17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        self.conv1 = nn.Conv2d(
            n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(2048, num_classes)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        return logits


class ResNet101(nn.Module):
    """ResNet101 implementation"""
    def __init__(self, n_inputs=12, num_classes=17):
        super().__init__()

        resnet = models.resnet101(pretrained=False)

        self.conv1 = nn.Conv2d(
            n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(2048, num_classes)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        return logits


class ResNet152(nn.Module):
    """ResNet152 implementation"""
    def __init__(self, n_inputs=12, num_classes=17):
        super().__init__()

        resnet = models.resnet152(pretrained=False)

        self.conv1 = nn.Conv2d(
            n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(2048, num_classes)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        return logits


if __name__ == "__main__":

    inputs = torch.randn(
        (1, 12, 256, 256)
    )  # (how many images, spectral channels, pxl, pxl)

    net = ResNet18()
    # net = ResNet34()
    # net = ResNet50()
    # net = ResNet101()
    # net = ResNet152()

    outputs = net(inputs)

    print(outputs)
    print(outputs.shape)

    numParams = count_parameters(net)

    print(f"{numParams:.2E}")
