import torch
import torch.nn as nn
import numpy as np

#  TODO:
# copied and altered to expose weights from the pytorch source code, modeled from vgg11
# simplified the VGG11 network greatly to reduce training for the toy example
class VGGToy(nn.Module):

    def __init__(self, device, num_classes=10, init_weights=True, batch_norm=False, noise_injection=False):
        super(VGGToy, self).__init__()


        self.device = device
        self.num_conv_layers = 3
        self.batch_norm = batch_norm
        self.noise_injection = noise_injection

        self.layer_inputs = []

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def _add_noise(self, x):
        mean1 = x.mean().item()
        std1 = x.std().item() * np.random.uniform(1, 2)
        noise1 = torch.tensor(np.random.normal(loc=mean1, scale=std1, size=x.shape)).to(self.device)
        with torch.no_grad():
            x += noise1.detach()

        return x

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        if self.noise_injection:
            x = self._add_noise(x)
        # only store 1 pass of weights
        self.layer_inputs = [x.detach()]
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.noise_injection:
            x = self._add_noise(x)
        self.layer_inputs.append(x.detach())
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        if self.noise_injection:
            x = self._add_noise(x)
        self.layer_inputs.append(x.detach())
        x = self.activation(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)